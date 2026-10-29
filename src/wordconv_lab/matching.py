"""Trainable context-to-source-span matching with an explicit Q E^T score.

All fitting data selection and experiment splits belong to the caller.  This
module never opens a corpus and accepts only exact VisibleQuery objects.  The
candidate universe is fixed at construction.  Every candidate participates in
the full-softmax loss; an absent original target has no legal training index.

Character embeddings are shared by a positional query CNN and an ordered
two-character span encoder.  Smooth L2-normalized vectors have a fixed
temperature.  There are no candidate-ID embeddings or independent output
biases: scores = (normalized_query / temperature) @ normalized_spans.T.
This is a source-restoration model, not a literary-quality or prosody oracle.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from numbers import Integral,Real
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np

from .restoration import VisibleQuery,digest


def _positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _positive_finite(value,name,*,zero_allowed=False):
    if (isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value)
            or value<0 or (not zero_allowed and value==0)):
        raise ValueError(name+' must be finite and '+('nonnegative' if zero_allowed else 'positive'))
    return float(value)


def _readonly(array):
    array.setflags(write=False)
    return array


class ContextSpanMatcher:
    """Small NumPy dual encoder with full-vocabulary cross-entropy training.

    Query positions are two separate seven-slot rows. Lower positions 0 and 1
    are unavailable; its suffix starts at position 2. Missing positions use -1
    plus a separate Boolean mask, while a real unknown character has ID 0 and
    remains visible. The query projection concatenates upper positions 0..6
    and lower positions 2..6 in this fixed order.

    context_mode is a training-arm definition. Inference mask is an additional
    intervention: 'upper' removes the upper context, 'suffix' removes the lower
    suffix, and 'both' removes both. Removed contexts never reach the CNN.
    """
    format_version='versematrix.context-span-matcher.v1'
    padding_id=-1

    def __init__(self,vocabulary,candidates_by_length,*,embedding_dim=12,
                 representation_dim=16,kernel_width=3,channels=24,
                 context_mode='both',temperature=.1,norm_epsilon=1e-6,seed=0):
        self.embedding_dim=_positive_integer(embedding_dim,'embedding_dim')
        self.representation_dim=_positive_integer(representation_dim,'representation_dim')
        self.channels=_positive_integer(channels,'channels')
        if isinstance(kernel_width,bool) or kernel_width not in (1,3):
            raise ValueError('kernel_width must be 1 or 3')
        if context_mode not in ('both','upper','suffix'):
            raise ValueError('context_mode must be both, upper or suffix')
        if isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0:
            raise ValueError('seed must be a nonnegative integer')
        self.kernel_width=int(kernel_width)
        self.context_mode=context_mode
        self.temperature=_positive_finite(temperature,'temperature')
        self.norm_epsilon=_positive_finite(norm_epsilon,'norm_epsilon')
        self.seed=int(seed)

        vocabulary=dict(vocabulary)
        if (not vocabulary or vocabulary.get('<UNK>')!=0
                or any(not isinstance(token,str) or (token!='<UNK>' and len(token)!=1)
                       or isinstance(index,bool) or not isinstance(index,Integral)
                       for token,index in vocabulary.items())
                or sorted(vocabulary.values())!=list(range(len(vocabulary)))):
            raise ValueError('vocabulary must be consecutive character IDs with explicit <UNK> at zero')
        self.vocabulary=MappingProxyType({token:int(index) for token,index in vocabulary.items()})
        self.vocabulary_tokens=tuple(token for token,_ in sorted(self.vocabulary.items(),key=lambda item:item[1]))
        self.vocab_size=len(vocabulary)
        candidates={}
        for length,spans in dict(candidates_by_length).items():
            if isinstance(length,bool) or str(length) not in ('5','7'):
                raise ValueError('candidate pools must be keyed by 5 or 7')
            length=int(length)
            if length in candidates:
                raise ValueError('duplicate normalized candidate-pool key')
            spans=tuple(spans)
            if (not spans or len(set(spans))!=len(spans)
                    or any(not isinstance(span,str) or len(span)!=2 for span in spans)):
                raise ValueError('each candidate pool must contain distinct complete two-character spans')
            if any(character not in self.vocabulary for span in spans for character in span):
                raise ValueError('candidate characters must exist in the fixed training vocabulary')
            candidates[length]=spans
        if not candidates:
            raise ValueError('at least one fixed candidate pool is required')
        self.candidates_by_length=MappingProxyType(dict(sorted(candidates.items())))
        self.candidate_indices=MappingProxyType({length:MappingProxyType({span:i for i,span in enumerate(spans)})
                                                for length,spans in self.candidates_by_length.items()})
        self._candidate_ids={length:_readonly(np.asarray([[self.vocabulary[c] for c in span] for span in spans],dtype=np.int64))
                             for length,spans in self.candidates_by_length.items()}
        self.config=MappingProxyType({
            'embedding_dim':self.embedding_dim,'representation_dim':self.representation_dim,
            'kernel_width':self.kernel_width,'channels':self.channels,'context_mode':self.context_mode,
            'temperature':self.temperature,'norm_epsilon':self.norm_epsilon,'seed':self.seed,
        })

        def rng(name):
            # Parameter-specific seeds keep all shared shapes identically
            # initialized across architecture/context arms with the same seed.
            key=hashlib.sha256(f'versematrix-matcher-v1:{self.seed}:{name}'.encode()).digest()
            return np.random.default_rng(int.from_bytes(key[:16],'big'))

        def xavier(name,shape,fan_in,fan_out):
            return rng(name).normal(0.0,math.sqrt(2.0/(fan_in+fan_out)),shape)

        e,d,c,k=self.embedding_dim,self.representation_dim,self.channels,self.kernel_width
        self.params={
            'embedding':rng('embedding').normal(0.0,.1,(self.vocab_size,e)),
            'context_kernel':xavier('context_kernel',(k,e,c),k*e,c),
            'context_bias':np.zeros(c,dtype=np.float64),
            'query_projection':xavier('query_projection',(12*c,d),12*c,d),
            'query_bias':np.zeros(d,dtype=np.float64),
            'span_projection':xavier('span_projection',(2*e,d),2*e,d),
            'span_projection_bias':np.zeros(d,dtype=np.float64),
        }
        self._trainable_masks={name:np.ones(array.shape,dtype=np.bool_) for name,array in self.params.items()}
        self._trainable_masks['embedding'][0]=False
        if context_mode=='upper':
            self._trainable_masks['query_projection'][7*c:]=False
        elif context_mode=='suffix':
            self._trainable_masks['query_projection'][:7*c]=False
        for name,array in self.params.items():
            array[~self._trainable_masks[name]]=0.0
        self.adam_m={name:np.zeros_like(array) for name,array in self.params.items()}
        self.adam_v={name:np.zeros_like(array) for name,array in self.params.items()}
        self.step_count=0
        self.last_optimizer=None

    def _shapes(self):
        e,d,c=self.embedding_dim,self.representation_dim,self.channels
        return {'embedding':(self.vocab_size,e),'context_kernel':(self.kernel_width,e,c),
                'context_bias':(c,),'query_projection':(12*c,d),'query_bias':(d,),
                'span_projection':(2*e,d),'span_projection_bias':(d,)}

    def _check_parameters(self,*,optimizer=False):
        groups=[('parameter',self.params)]
        if optimizer:
            groups.extend((('Adam first moment',self.adam_m),('Adam second moment',self.adam_v)))
        shapes=self._shapes()
        for label,values in groups:
            if set(values)!=set(shapes):
                raise ValueError('invalid '+label+' names')
            for name,shape in shapes.items():
                array=values[name]
                if (not isinstance(array,np.ndarray) or array.dtype!=np.dtype(np.float64)
                        or array.shape!=shape or not np.isfinite(array).all()):
                    raise ValueError('invalid finite '+label+': '+name)
                if np.any(array[~self._trainable_masks[name]]!=0.0):
                    raise ValueError('frozen UNK/inactive '+label+' must remain exactly zero: '+name)
        if optimizer and any(np.any(array<0) for array in self.adam_v.values()):
            raise ValueError('Adam second moments cannot be negative')
        if isinstance(self.step_count,bool) or not isinstance(self.step_count,Integral) or self.step_count<0:
            raise ValueError('invalid optimizer update count')
        if self.step_count==0 and self.last_optimizer is not None:
            raise ValueError('untrained state must not report an optimizer update')
        if self.step_count>0 and not isinstance(self.last_optimizer,dict):
            raise ValueError('trained state must record its last optimizer configuration')
        if optimizer and self.step_count>0:
            expected={'name','learning_rate','weight_decay','max_grad_norm','beta1','beta2','epsilon'}
            if (set(self.last_optimizer)!=expected
                    or self.last_optimizer['name']!='Adam_with_decoupled_weight_decay'
                    or self.last_optimizer['beta1']!=.9 or self.last_optimizer['beta2']!=.999
                    or self.last_optimizer['epsilon']!=1e-8):
                raise ValueError('optimizer metadata differs from the declared Adam contract')
            _positive_finite(self.last_optimizer['learning_rate'],'saved learning_rate')
            _positive_finite(self.last_optimizer['weight_decay'],'saved weight_decay',zero_allowed=True)
            if self.last_optimizer['max_grad_norm'] is not None:
                _positive_finite(self.last_optimizer['max_grad_norm'],'saved max_grad_norm')

    def _queries(self,queries):
        try:
            queries=tuple(queries)
        except TypeError as exc:
            raise ValueError('queries must be a nonempty same-length VisibleQuery batch') from exc
        if not queries or any(type(query) is not VisibleQuery for query in queries):
            raise ValueError('only a nonempty batch of exact VisibleQuery objects is accepted')
        length=queries[0].length
        if any(query.length!=length for query in queries) or length not in self.candidates_by_length:
            raise ValueError('queries must have one length with a fixed candidate pool')
        return queries,length

    def encode_queries(self,queries,*,mask='none'):
        queries,length=self._queries(queries)
        if mask not in ('none','upper','suffix','both'):
            raise ValueError('inference mask must be none, upper, suffix or both')
        ids=np.full((len(queries),2,7),self.padding_id,dtype=np.int64)
        for row,query in enumerate(queries):
            ids[row,0,:length]=[self.vocabulary.get(character,0) for character in query.upper_line]
            ids[row,1,2:length]=[self.vocabulary.get(character,0) for character in query.lower_suffix]
        visible=ids!=self.padding_id
        active=visible.copy()
        if self.context_mode=='suffix' or mask in ('upper','both'):
            active[:,0]=False
        if self.context_mode=='upper' or mask in ('suffix','both'):
            active[:,1]=False
        return {'token_ids':_readonly(ids),'visible_mask':_readonly(visible),
                'active_mask':_readonly(active),'length':length,
                'unknown_visible_characters':int(np.count_nonzero((ids==0)&visible)),
                'unknown_active_characters':int(np.count_nonzero((ids==0)&active))}

    def _normalize(self,raw):
        denominator=np.sqrt(np.sum(raw*raw,axis=-1,keepdims=True)+self.norm_epsilon*self.norm_epsilon)
        return raw/denominator,denominator

    @staticmethod
    def _normalization_backward(gradient,normalized,denominator):
        return (gradient-normalized*np.sum(gradient*normalized,axis=-1,keepdims=True))/denominator

    def _query_forward(self,queries,mask='none',*,keep_cache=False):
        encoded=self.encode_queries(queries,mask=mask)
        active=encoded['active_mask']
        lookup_ids=np.where(active,encoded['token_ids'],0)
        x=self.params['embedding'][lookup_ids]*active[...,None]
        radius=self.kernel_width//2
        padded=np.pad(x,((0,0),(0,0),(radius,radius),(0,0)))
        pre=np.zeros((len(lookup_ids),2,7,self.channels),dtype=np.float64)
        pre+=self.params['context_bias']
        for offset in range(self.kernel_width):
            pre+=padded[:,:,offset:offset+7,:]@self.params['context_kernel'][offset]
        activated=np.tanh(pre)
        hidden=activated*active[...,None]
        joined=np.concatenate((hidden[:,0],hidden[:,1,2:7]),axis=1).reshape(len(lookup_ids),12*self.channels)
        raw=np.tanh(joined@self.params['query_projection']+self.params['query_bias'])
        normalized,denominator=self._normalize(raw)
        cache=(lookup_ids,active,padded,activated,joined,raw,normalized,denominator) if keep_cache else None
        return normalized,cache,encoded['length']

    def _candidate_forward(self,length,*,keep_cache=False):
        if isinstance(length,bool) or not isinstance(length,Integral) or int(length) not in self._candidate_ids:
            raise ValueError('length must identify a fixed candidate pool')
        ids=self._candidate_ids[int(length)]
        joined=self.params['embedding'][ids].reshape(len(ids),2*self.embedding_dim)
        raw=np.tanh(joined@self.params['span_projection']+self.params['span_projection_bias'])
        normalized,denominator=self._normalize(raw)
        cache=(ids,joined,raw,normalized,denominator) if keep_cache else None
        return normalized,cache

    def _forward(self,queries,mask='none',*,keep_cache=False):
        self._check_parameters()
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                q,qcache,length=self._query_forward(queries,mask,keep_cache=keep_cache)
                e,ecache=self._candidate_forward(length,keep_cache=keep_cache)
                scaled_q=q/self.temperature
                scores=scaled_q@e.T
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in matching forward pass') from exc
        if not all(np.isfinite(array).all() for array in (q,e,scaled_q,scores)):
            raise ValueError('nonfinite matching factor or score')
        return scores,scaled_q,e,(qcache,ecache) if keep_cache else None

    def score_all(self,queries,*,mask='none'):
        """Full fixed-vocabulary scores for one same-length query batch."""
        return self._forward(queries,mask)[0]

    def query_embeddings(self,queries,*,mask='none'):
        """Smooth normalized query vectors, before fixed temperature scaling."""
        self._check_parameters()
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                q,_,_=self._query_forward(queries,mask)
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in query encoder') from exc
        if not np.isfinite(q).all():
            raise ValueError('nonfinite query representation')
        return q

    def candidate_embeddings(self,length):
        """Smooth normalized ordered-span vectors, rebuilt from current weights.

        No candidate-encoding cache survives a parameter update or manual audit
        perturbation; both encoders always use the same current embedding table.
        """
        self._check_parameters()
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                e,_=self._candidate_forward(length)
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in candidate encoder') from exc
        if not np.isfinite(e).all():
            raise ValueError('nonfinite candidate representation')
        return e

    def score_factors(self,queries,*,mask='none'):
        """Audit factors satisfy scores == Q @ E.T + bias (bias is zero)."""
        scores,q,e,_=self._forward(queries,mask)
        return {'Q':q,'E':e,'bias':np.zeros(e.shape[0],dtype=np.float64),'scores':scores,
                'temperature':self.temperature,'bias_trainable':False}

    def rank(self,query,*,limit=16,mask='none'):
        if type(query) is not VisibleQuery:
            raise ValueError('rank accepts one exact VisibleQuery')
        if isinstance(limit,bool) or not isinstance(limit,Integral) or limit<1:
            raise ValueError('limit must be a positive integer')
        scores=self.score_all([query],mask=mask)[0]
        spans=self.candidates_by_length[query.length]
        order=np.lexsort((np.asarray(spans,dtype=str),-scores))[:int(limit)]
        exponential=np.exp(scores-scores.max())
        probabilities=exponential/exponential.sum()
        return {'candidates':tuple(spans[int(i)] for i in order),
                'scores':[float(scores[i]) for i in order],
                'model_probabilities':[float(probabilities[i]) for i in order],
                'query_hash':digest(asdict(query)),'inference_mask':mask,
                'candidate_count':len(spans),
                'interpretation':'full-vocabulary source-restoration model probabilities, not literary-quality confidence'}

    def loss_and_gradients(self,queries,target_indices,*,mask='none'):
        """Mean full-softmax NLL and manual derivatives of every shared weight.

        An original target outside the fixed vocabulary cannot be represented by
        this API. Callers must omit such items from training and keep them as
        zero-recall evaluation items; this method never inserts new candidates.
        The reported loss excludes optimizer-side decoupled weight decay.
        """
        scores,scaled_q,e,caches=self._forward(queries,mask,keep_cache=True)
        batch,choices=scores.shape
        targets=np.asarray(target_indices)
        if (targets.shape!=(batch,) or targets.dtype.kind not in 'iu'
                or np.any(targets<0) or np.any(targets>=choices)):
            raise ValueError('one integer target index inside the complete fixed pool is required per query')
        qcache,ecache=caches
        lookup_ids,active,padded,activated,joined_q,raw_q,normalized_q,denom_q=qcache
        candidate_ids,joined_e,raw_e,normalized_e,denom_e=ecache
        gradients={name:np.zeros_like(array) for name,array in self.params.items()}
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                centered=scores-scores.max(axis=1,keepdims=True)
                exponential=np.exp(centered)
                denominator=exponential.sum(axis=1,keepdims=True)
                loss=float(np.mean(np.log(denominator[:,0])-centered[np.arange(batch),targets]))
                ds=exponential/denominator
                ds[np.arange(batch),targets]-=1.0
                ds/=batch
                dq=(ds@e)/self.temperature
                de=ds.T@scaled_q
                dzq=self._normalization_backward(dq,normalized_q,denom_q)*(1.0-raw_q*raw_q)
                dze=self._normalization_backward(de,normalized_e,denom_e)*(1.0-raw_e*raw_e)
                gradients['query_projection']=joined_q.T@dzq
                gradients['query_bias']=dzq.sum(axis=0)
                gradients['span_projection']=joined_e.T@dze
                gradients['span_projection_bias']=dze.sum(axis=0)
                djoined_q=(dzq@self.params['query_projection'].T).reshape(batch,12,self.channels)
                dh=np.zeros_like(activated)
                dh[:,0]=djoined_q[:,:7]
                dh[:,1,2:7]=djoined_q[:,7:]
                dpre=dh*active[...,None]*(1.0-activated*activated)
                gradients['context_bias']=dpre.sum(axis=(0,1,2))
                dx_padded=np.zeros_like(padded)
                dpre_flat=dpre.reshape(-1,self.channels)
                for offset in range(self.kernel_width):
                    window=padded[:,:,offset:offset+7,:]
                    gradients['context_kernel'][offset]=window.reshape(-1,self.embedding_dim).T@dpre_flat
                    dx_padded[:,:,offset:offset+7,:]+=dpre@self.params['context_kernel'][offset].T
                radius=self.kernel_width//2
                dx=dx_padded[:,:,radius:radius+7,:]*active[...,None]
                np.add.at(gradients['embedding'],lookup_ids.reshape(-1),dx.reshape(-1,self.embedding_dim))
                djoined_e=(dze@self.params['span_projection'].T).reshape(-1,self.embedding_dim)
                np.add.at(gradients['embedding'],candidate_ids.reshape(-1),djoined_e)
                for name,gradient in gradients.items():
                    gradient[~self._trainable_masks[name]]=0.0
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in full-softmax matching loss or gradient') from exc
        if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
            raise ValueError('nonfinite matching loss or gradient')
        return loss,gradients

    def train_step(self,queries,target_indices,*,learning_rate=.003,weight_decay=.0001,max_grad_norm=5.0):
        """One atomic Adam update, with fixed beta1/beta2/epsilon and full softmax."""
        # Materialize one-shot iterables before any update can commit. The same
        # validated tuple supplies both the loss and the eventual receipt.
        queries,length=self._queries(queries)
        learning_rate=_positive_finite(learning_rate,'learning_rate')
        weight_decay=_positive_finite(weight_decay,'weight_decay',zero_allowed=True)
        if max_grad_norm is not None:
            max_grad_norm=_positive_finite(max_grad_norm,'max_grad_norm')
        self._check_parameters(optimizer=True)
        loss,gradients=self.loss_and_gradients(queries,target_indices)
        step=self.step_count+1
        new_params,new_m,new_v={},{},{}
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                norm=math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
                scale=min(1.0,max_grad_norm/norm) if max_grad_norm is not None and norm>0 else 1.0
                for name,old in self.params.items():
                    gradient=gradients[name]*scale
                    m=.9*self.adam_m[name]+.1*gradient
                    v=.999*self.adam_v[name]+.001*gradient*gradient
                    corrected_m=m/(1.0-.9**step)
                    corrected_v=v/(1.0-.999**step)
                    updated=old-learning_rate*(corrected_m/(np.sqrt(corrected_v)+1e-8)+weight_decay*old)
                    frozen=~self._trainable_masks[name]
                    updated[frozen]=0.0
                    m[frozen]=0.0
                    v[frozen]=0.0
                    if not all(np.isfinite(array).all() for array in (updated,m,v)):
                        raise ValueError('nonfinite Adam parameter or state')
                    new_params[name],new_m[name],new_v[name]=updated,m,v
        except FloatingPointError as exc:
            raise ValueError('nonfinite arithmetic in matching Adam update') from exc
        if not math.isfinite(norm):
            raise ValueError('nonfinite gradient norm')
        self.params,self.adam_m,self.adam_v=new_params,new_m,new_v
        self.step_count=step
        self.last_optimizer={'name':'Adam_with_decoupled_weight_decay','learning_rate':learning_rate,
                             'weight_decay':weight_decay,'max_grad_norm':max_grad_norm,
                             'beta1':.9,'beta2':.999,'epsilon':1e-8}
        return {'loss':loss,'gradient_norm':norm,'clip_scale':scale,'step':step,
                'batch_size':len(queries),'full_softmax_candidates':len(self.candidates_by_length[length])}

    def parameter_count(self):
        return sum(array.size for array in self.params.values())

    def trainable_parameter_count(self):
        return sum(int(mask.sum()) for mask in self._trainable_masks.values())

    def parameter_summary(self):
        stored=self.parameter_count()
        trainable=self.trainable_parameter_count()
        return {'stored_parameters':stored,'trainable_parameters':trainable,
                'frozen_parameters':stored-trainable,'frozen_unk_parameters':self.embedding_dim,
                'frozen_inactive_query_projection_parameters':int(np.count_nonzero(~self._trainable_masks['query_projection'])),
                'adam_state_scalars_allocated':2*stored,'step_count':self.step_count,
                'by_tensor':{name:{'shape':list(array.shape),'stored':array.size,
                                   'trainable':int(self._trainable_masks[name].sum())}
                             for name,array in self.params.items()},
                'candidate_id_embeddings':False,'independent_candidate_bias':False,
                'output_bias':'identically_zero','representation_dim':self.representation_dim,
                'temperature':self.temperature,'norm_epsilon':self.norm_epsilon,
                'context_mode':self.context_mode,'dtype':'float64'}

    def _metadata(self):
        pools={str(length):list(spans) for length,spans in self.candidates_by_length.items()}
        return {'format':self.format_version,'config':dict(self.config),
                'vocabulary_tokens':list(self.vocabulary_tokens),'candidates_by_length':pools,
                'vocabulary_sha256':digest(self.vocabulary_tokens),'candidates_sha256':digest(pools),
                'step_count':int(self.step_count),'last_optimizer':self.last_optimizer,
                'padding_id':self.padding_id,'output_bias':'identically_zero',
                'interpretation':'source_restoration_only_no_literary_quality_or_prosody_certification'}

    def save(self,path):
        """Atomically create a new checkpoint; an existing path is never replaced."""
        self._check_parameters(optimizer=True)
        path=Path(path)
        if path.exists():
            raise FileExistsError('checkpoint already exists: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays={'metadata':np.asarray(json.dumps(self._metadata(),ensure_ascii=False,sort_keys=True,allow_nan=False))}
        for prefix,collection in (('parameter',self.params),('adam_m',self.adam_m),('adam_v',self.adam_v)):
            arrays.update({prefix+'_'+name:array for name,array in collection.items()})
        descriptor,temporary_name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
        temporary=Path(temporary_name)
        try:
            with os.fdopen(descriptor,'wb') as handle:
                np.savez_compressed(handle,**arrays)
                handle.flush()
                os.fsync(handle.fileno())
            # Same-directory hard link is an atomic no-replace publication.
            # os.replace would be unsafe if another writer created path after
            # the initial existence check.
            os.link(temporary,path)
            directory_fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        data=path.read_bytes()
        return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'step_count':int(self.step_count),'write_policy':'atomic_create_only'}

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata'].item()))
            if (metadata.get('format')!=cls.format_version or metadata.get('padding_id')!=cls.padding_id
                    or metadata.get('output_bias')!='identically_zero'):
                raise ValueError('checkpoint format or scoring contract mismatch')
            tokens=metadata['vocabulary_tokens']
            pools=metadata['candidates_by_length']
            if (digest(tuple(tokens))!=metadata['vocabulary_sha256']
                    or digest(pools)!=metadata['candidates_sha256']):
                raise ValueError('checkpoint vocabulary/candidate identity mismatch')
            if len(set(tokens))!=len(tokens):
                raise ValueError('duplicate checkpoint vocabulary token')
            model=cls({token:i for i,token in enumerate(tokens)},pools,**metadata['config'])
            expected={'metadata'}|{prefix+'_'+name for prefix in ('parameter','adam_m','adam_v') for name in model.params}
            if set(archive.files)!=expected:
                raise ValueError('checkpoint tensor names differ from declared architecture')
            for prefix,attribute in (('parameter','params'),('adam_m','adam_m'),('adam_v','adam_v')):
                setattr(model,attribute,{name:np.array(archive[prefix+'_'+name],copy=True) for name in model.params})
            model.step_count=metadata['step_count']
            model.last_optimizer=metadata['last_optimizer']
        model._check_parameters(optimizer=True)
        return model
