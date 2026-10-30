#!/usr/bin/env python3
"""Recheck output receipts against training bytes and reference sets, not quality."""
import argparse, collections, csv, hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.prosody import all_templates, variants

def check(ok,message):
    if not ok: raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_induced_model.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    poems,_=b.training_files(ROOT,a.bootstrap);byid={p['poem_id']:p for p in poems};lines={s for p in poems for s in p['lines']}
    data=json.loads((ROOT/'web/induced-data.js').read_text().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]);words=data['lexicon']
    source_records=list(csv.DictReader((a.bootstrap/'cache/raw/guangyun.csv').open(encoding='utf-8-sig')))
    ref=json.loads((ROOT/'data/reference_v02/pingshui_106_memberships.json').read_bytes())['characters']
    count=0;completed=0;positions=0;receipts=0
    for raw in a.input.read_text().splitlines():
        r=json.loads(raw);count+=1
        if r['status']!='complete':check(r['lines']==[],'Failure contains a poem');continue
        completed+=1;n=r['length'];check(len(r['lines'])==8,'Not eight lines')
        template=next(t for t in all_templates(n) if t.template_id==r['templateId']);seen=collections.Counter();rhymes=[]
        check(r['provenance']['preselectedLines'] is False,'Preselected line claim')
        for row,(text,detail) in enumerate(zip(r['lines'],r['lineDetails'])):
            check(len(text)==n and text not in lines,'Shape or training-line copying');check(len(set(text))==n,'Within-line repeats');seen.update(text)
            check(''.join(t['text'] for t in detail['tokens'])==text,'Token reconstruction mismatch');tone=''
            for t in detail['tokens']:
                base=words[t['id']];check(t['text']==base['text'],'Unknown token');pid=t['sourcePoem']['poemId'];check(pid in byid,'Not training')
                _,source_row,start,end=t['source'];check(byid[pid]['lines'][source_row][start:end]==t['text'],'False source substring')
                for rc in t['readings']:
                    c=rc['character'];check(rc['selectedEntryId'] is None and not rc['contextuallyCertified'],'False contextual certification')
                    indices=[int(i.split(':')[1])-1 for i in rc['candidateEntryIds']];check(all(source_records[i]['字頭']==c for i in indices),'Wrong historical headword')
                    tones={'P' if source_records[i]['音韻地位'].endswith('平') else 'Z' for i in indices}
                    check(tones=={rc['tone']}=={e['tone'] for e in ref[c]},'Ambiguous tone hidden');tone+=rc['tone'];positions+=1
            check(tone==detail['tones'] and tone in variants(template.core[row]),'Meter mismatch')
            if row in template.rhyme_rows:
                groups={m['group_id'] for m in ref[text[-1]] if m['tone']=='P'};check(groups=={r['rhymeGroup']},'Rhyme mismatch');rhymes.append(text[-1])
            for edge in detail['transitions']:
                if edge['from'] is not None:
                    table={v:c for v,c,_ in data['next'].get(str(edge['from']),[])}
                    check(edge['count']==table.get(edge['to'],0),'Incorrect adjacency receipt')
                receipts+=1
        check(max(seen.values())<=2 and len(set(rhymes))==len(rhymes),'Repetition constraint')
    out={'status':'PASS_RECEIPTS_AND_DECLARED_CONSTRAINTS','attempted':count,'completed':completed,'failed':count-completed,'characterPositions':positions,'transitionReceipts':receipts,'strictAntithesisCertified':False,'literaryQualityCertified':False,'contextualPronunciationCertified':False,'inputSha256':hashlib.sha256(a.input.read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
