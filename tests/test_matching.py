import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wordconv_lab.matching import ContextSpanMatcher
from wordconv_lab.restoration import VisibleQuery


VOCAB={character:index for index,character in enumerate(['<UNK>']+list('甲乙丙丁戊己庚'))}
POOLS={5:('甲乙','乙甲','丙甲','甲甲'),7:('甲乙','乙丙','丙甲','乙甲')}


def queries(length=5):
    if length==5:
        return [VisibleQuery(5,'甲乙甲丁戊','丙己庚'),VisibleQuery(5,'乙甲丁甲丙','甲乙甲')]
    return [VisibleQuery(7,'甲乙甲丁戊己庚','丙己庚甲乙'),
            VisibleQuery(7,'乙甲丁甲丙戊己','甲乙甲乙丙')]


def small_model(**options):
    settings={'embedding_dim':3,'representation_dim':4,'channels':2,'seed':417}
    settings.update(options)
    return ContextSpanMatcher(VOCAB,POOLS,**settings)


class ContextSpanMatchingTests(unittest.TestCase):
    def test_normalized_shared_encoder_gradients_match_central_differences(self):
        targets=np.array([0,2],dtype=np.int64)
        for width in (1,3):
            for length in (5,7):
                model=small_model(kernel_width=width)
                batch=queries(length)
                _,gradients=model.loss_and_gradients(batch,targets)
                coordinates={
                    'embedding':[(1,0),(2,1),(6,2)],
                    'context_kernel':[(0,0,0),(width-1,2,1)],
                    'context_bias':[(0,),(1,)],
                    'query_projection':[(0,0),(14,1),(23,3)],
                    'query_bias':[(0,),(3,)],
                    'span_projection':[(0,0),(4,2)],
                    'span_projection_bias':[(1,),(3,)],
                }
                for name,indices in coordinates.items():
                    for index in indices:
                        with self.subTest(width=width,length=length,name=name,index=index):
                            original=model.params[name][index]
                            epsilon=1e-6
                            model.params[name][index]=original+epsilon
                            plus=model.loss_and_gradients(batch,targets)[0]
                            model.params[name][index]=original-epsilon
                            minus=model.loss_and_gradients(batch,targets)[0]
                            model.params[name][index]=original
                            numerical=(plus-minus)/(2*epsilon)
                            self.assertAlmostEqual(gradients[name][index],numerical,delta=3e-6)

    def test_explicit_qe_factors_batch_single_and_full_softmax_oracle(self):
        for length in (5,7):
            model=small_model()
            batch=queries(length)
            factors=model.score_factors(batch)
            np.testing.assert_array_equal(factors['scores'],factors['Q']@factors['E'].T+factors['bias'])
            np.testing.assert_array_equal(factors['bias'],np.zeros(len(POOLS[length])))
            np.testing.assert_allclose(model.score_all(batch),np.vstack([model.score_all([q])[0] for q in batch]),
                                       rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(factors['Q'],model.query_embeddings(batch)/model.temperature,rtol=0,atol=0)
            np.testing.assert_array_equal(factors['E'],model.candidate_embeddings(length))
            targets=np.array([0,2])
            expected=np.mean([np.logaddexp.reduce(row)-row[target]
                              for row,target in zip(factors['scores'],targets)])
            self.assertAlmostEqual(model.loss_and_gradients(batch,targets)[0],expected,places=13)

    def test_ordered_candidate_characters_and_pool_permutation_are_respected(self):
        model=small_model()
        embeddings=model.candidate_embeddings(5)
        self.assertGreater(np.linalg.norm(embeddings[0]-embeddings[1]),1e-5)
        order=(2,0,3,1)
        reversed_pool={5:tuple(POOLS[5][i] for i in order),7:POOLS[7]}
        other=ContextSpanMatcher(VOCAB,reversed_pool,**dict(model.config))
        batch=queries()
        targets=np.array([0,2])
        remapped=np.array([order.index(int(target)) for target in targets])
        np.testing.assert_allclose(other.score_all(batch),model.score_all(batch)[:,order],rtol=1e-12,atol=1e-12)
        loss,gradient=model.loss_and_gradients(batch,targets)
        permuted,permuted_gradient=other.loss_and_gradients(batch,remapped)
        self.assertAlmostEqual(loss,permuted,places=13)
        for name in gradient:
            np.testing.assert_allclose(gradient[name],permuted_gradient[name],rtol=1e-11,atol=1e-11)

    def test_padding_is_not_unknown_and_frozen_values_stay_zero_after_updates(self):
        model=small_model(context_mode='upper')
        batch=[VisibleQuery(5,'龘乙甲丁戊','丙己庚'),queries()[1]]
        encoded=model.encode_queries(batch)
        self.assertEqual(encoded['token_ids'][0,0,0],0)
        self.assertTrue(encoded['visible_mask'][0,0,0])
        self.assertTrue(encoded['active_mask'][0,0,0])
        np.testing.assert_array_equal(encoded['token_ids'][:,1,:2],np.full((2,2),-1))
        self.assertFalse(encoded['visible_mask'][:,1,:2].any())
        self.assertFalse(encoded['active_mask'][:,1].any())
        _,gradient=model.loss_and_gradients(batch,np.array([0,2]))
        np.testing.assert_array_equal(gradient['embedding'][0],np.zeros(3))
        np.testing.assert_array_equal(gradient['query_projection'][14:],np.zeros((10,4)))
        for _ in range(5):
            model.train_step(batch,np.array([0,2]),weight_decay=.1)
        for collection in (model.params,model.adam_m,model.adam_v):
            np.testing.assert_array_equal(collection['embedding'][0],np.zeros(3))
            np.testing.assert_array_equal(collection['query_projection'][14:],np.zeros((10,4)))

    def test_training_context_arms_and_inference_masks_remove_the_named_information(self):
        original=VisibleQuery(5,'甲乙丙丁戊','丙己庚')
        changed_upper=VisibleQuery(5,'己庚甲乙丙','丙己庚')
        changed_suffix=VisibleQuery(5,'甲乙丙丁戊','甲乙甲')
        upper=small_model(context_mode='upper')
        suffix=small_model(context_mode='suffix')
        both=small_model()
        np.testing.assert_array_equal(upper.score_all([original]),upper.score_all([changed_suffix]))
        np.testing.assert_array_equal(suffix.score_all([original]),suffix.score_all([changed_upper]))
        np.testing.assert_array_equal(both.score_all([original],mask='upper'),both.score_all([changed_upper],mask='upper'))
        np.testing.assert_array_equal(both.score_all([original],mask='suffix'),both.score_all([changed_suffix],mask='suffix'))
        np.testing.assert_array_equal(both.score_all([original],mask='both'),both.score_all([changed_upper],mask='both'))
        self.assertFalse(np.allclose(both.score_all([original]),both.score_all([changed_upper])))
        self.assertFalse(np.allclose(both.score_all([original]),both.score_all([changed_suffix])))

    def test_full_softmax_training_reduces_actual_loss_and_updates_span_encoder(self):
        model=small_model()
        batch=[VisibleQuery(5,'甲甲甲甲甲','丁丁丁'),VisibleQuery(5,'乙乙乙乙乙','戊戊戊')]
        targets=np.array([0,1])
        before=model.loss_and_gradients(batch,targets)[0]
        old_candidates=model.candidate_embeddings(5).copy()
        for _ in range(60):
            receipt=model.train_step(batch,targets,learning_rate=.005)
        after=model.loss_and_gradients(batch,targets)[0]
        self.assertLess(after,before*.3)
        self.assertEqual(receipt['step'],60)
        self.assertEqual(receipt['full_softmax_candidates'],4)
        self.assertFalse(np.allclose(old_candidates,model.candidate_embeddings(5)))

    def test_atomic_checkpoint_restores_next_update_and_never_overwrites_existing_file(self):
        model=small_model()
        batch=queries(7);targets=np.array([0,2])
        model.train_step(batch,targets)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'update1.npz'
            receipt=model.save(path)
            self.assertEqual(receipt['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            before_bytes=path.read_bytes()
            restored=ContextSpanMatcher.load(path)
            np.testing.assert_array_equal(model.score_all(batch),restored.score_all(batch))
            self.assertEqual(dict(model.config),dict(restored.config))
            self.assertEqual(model.train_step(batch,targets),restored.train_step(batch,targets))
            for name in model.params:
                np.testing.assert_array_equal(model.params[name],restored.params[name])
                np.testing.assert_array_equal(model.adam_m[name],restored.adam_m[name])
                np.testing.assert_array_equal(model.adam_v[name],restored.adam_v[name])
            with self.assertRaises(FileExistsError):
                model.save(path)
            self.assertEqual(path.read_bytes(),before_bytes)
            self.assertEqual([p.name for p in Path(folder).iterdir()],['update1.npz'])

    def test_width_controls_share_initial_embeddings_and_span_encoder_and_disclose_counts(self):
        local=ContextSpanMatcher(VOCAB,POOLS,kernel_width=3,channels=24,seed=19)
        point=ContextSpanMatcher(VOCAB,POOLS,kernel_width=1,channels=27,seed=19)
        upper=ContextSpanMatcher(VOCAB,POOLS,kernel_width=3,channels=24,context_mode='upper',seed=19)
        suffix=ContextSpanMatcher(VOCAB,POOLS,kernel_width=3,channels=24,context_mode='suffix',seed=19)
        for name in ('embedding','span_projection','span_projection_bias'):
            np.testing.assert_array_equal(local.params[name],point.params[name])
            np.testing.assert_array_equal(local.params[name],upper.params[name])
        self.assertEqual(point.parameter_count()-local.parameter_count(),39)
        self.assertEqual(local.parameter_count()-local.trainable_parameter_count(),12)
        self.assertEqual(local.trainable_parameter_count()-upper.trainable_parameter_count(),1920)
        self.assertEqual(local.trainable_parameter_count()-suffix.trainable_parameter_count(),2688)
        self.assertFalse(local.parameter_summary()['candidate_id_embeddings'])
        self.assertFalse(local.parameter_summary()['independent_candidate_bias'])

    def test_target_metadata_never_changes_query_and_exact_ties_use_lexical_order(self):
        model=small_model()
        task={'length':5,'upper_line':'甲乙甲丁戊','lower_suffix':'丙己庚','start':0,'mask_length':2,
              'target':'甲乙','title':'one','author':'one'}
        changed={**task,'target':'乙甲','title':'two','author':'two','lower_line':'乙甲丙己庚'}
        self.assertEqual(model.rank(VisibleQuery.from_task(task)),model.rank(VisibleQuery.from_task(changed)))
        ranked=model.rank(VisibleQuery.from_task(task),mask='both')
        self.assertEqual(ranked['candidates'],tuple(sorted(POOLS[5])))
        np.testing.assert_array_equal(ranked['scores'],np.zeros(4))
        self.assertAlmostEqual(sum(ranked['model_probabilities']),1.0)

    def test_generator_batch_is_materialized_before_optimizer_commit(self):
        model=small_model()
        twin=small_model()
        targets=np.array([0,2])
        one=model.train_step((q for q in queries()),targets)
        two=twin.train_step(queries(),targets)
        self.assertEqual(one,two)
        self.assertEqual(one['batch_size'],2)
        self.assertEqual(one['step'],1)
        for name in model.params:
            np.testing.assert_array_equal(model.params[name],twin.params[name])

    def test_invalid_targets_queries_vocab_and_nonfinite_state_are_rejected(self):
        model=small_model()
        for targets in (np.array([0.,1.]),np.array([True,False]),np.array([-1,0]),np.array([0,4]),np.array([0])):
            with self.subTest(targets=targets),self.assertRaises(ValueError):
                model.loss_and_gradients(queries(),targets)
        for batch in ([],[{}],[queries()[0],queries(7)[0]]):
            with self.subTest(batch=batch),self.assertRaises(ValueError):
                model.score_all(batch)
        class HiddenTargetQuery(VisibleQuery):
            target='甲乙'
        with self.assertRaises(ValueError):
            model.score_all([HiddenTargetQuery(5,'甲乙丙丁戊','丙己庚')])
        with self.assertRaises(ValueError):
            ContextSpanMatcher(VOCAB,{5:('未知',)})
        with self.assertRaises(ValueError):
            ContextSpanMatcher({'甲':0},POOLS)
        with self.assertRaises(ValueError):
            model.score_all(queries(),mask='target')
        model.params['span_projection'][0,0]=math.nan
        with self.assertRaises(ValueError):
            model.score_all(queries())

    def test_failed_numeric_update_cannot_partially_commit_parameters_or_optimizer(self):
        model=small_model()
        before={name:array.copy() for name,array in model.params.items()}
        with self.assertRaisesRegex(ValueError,'Adam'):
            model.train_step(queries(),np.array([0,2]),learning_rate=1e308,weight_decay=1e308)
        self.assertEqual(model.step_count,0)
        self.assertIsNone(model.last_optimizer)
        for name in before:
            np.testing.assert_array_equal(model.params[name],before[name])
            np.testing.assert_array_equal(model.adam_m[name],np.zeros_like(before[name]))
            np.testing.assert_array_equal(model.adam_v[name],np.zeros_like(before[name]))


if __name__=='__main__':
    unittest.main()
