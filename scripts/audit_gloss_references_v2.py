#!/usr/bin/env python3
"""Trace the pinned Guangyun reference graph, without selecting contextual readings."""
import argparse,csv,hashlib,io,json
from pathlib import Path
from build_composition_v2 import gloss_trace,blob,GY_BLOB,GY_REV

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('Refusing overwrite')
 raw=(a.bootstrap/'cache/raw/guangyun.csv').read_bytes()
 if blob(raw)!=GY_BLOB:raise SystemExit('Source identity mismatch')
 rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));traces=[gloss_trace(rows,i) for i in range(len(rows))]
 import collections
 pilot_path=a.bootstrap/'pilot_drafts.json'
 if not pilot_path.exists():pilot_path=Path(__file__).resolve().parents[1]/'data/quality_bootstrap_v1/pilot_drafts.json'
 pilot=json.loads(pilot_path.read_text());samples=[]
 for record in pilot['records']:
  for note in record['reading_notes']:
   for eid in note['entry_ids']:
    i=int(eid.split(':')[1])-1;r=rows[i]
    samples.append({'taskId':record['task_id'],'character':r['字頭'],'entryId':eid,'position':r['音韻地位'],
        'fanqie':r['反切'],'originalGloss':r['釋義'],'trace':traces[i],
        'contextuallySelected':False,'readingsTransferredFromGlossTarget':False})
 tasks=[json.loads(l) for l in (a.bootstrap/'run/annotation_tasks_400.jsonl').read_text().splitlines()]
 report={'schema':'verseloom.gloss-reference-audit.v2','sourceRevision':GY_REV,'sourceSha256':hashlib.sha256(raw).hexdigest(),
  'entries':len(rows),'statuses':dict(collections.Counter(t['status'] for t in traces)),
  'unresolvedEntryIds':['gy:'+str(i+1) for i,t in enumerate(traces) if t['status']!='resolved'],
  'developmentTasksIndexed':len(tasks),'taskReadingPositions':sum(len(t['reading_candidates']) for t in tasks),
  'pilotEvidence':samples,'contextuallyResolvedPositions':0,'independentReviewedAnnotations':0,
  'note':'References only follow explicit 上/下 edges. Cycles remain unresolved. Gloss inheritance never changes the original phonological entry.'}
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['pilotEvidence','unresolvedEntryIds']},ensure_ascii=False))
if __name__=='__main__':main()
