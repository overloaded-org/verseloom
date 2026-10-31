#!/usr/bin/env python3
"""Build data/lexicon.json from repository-built JS tables, verified against the evaluated export."""
import argparse, collections, hashlib, json
from pathlib import Path


def build(web, dest):
    def table(name):
        s=(web/(name+'-data.js')).read_text(encoding='utf-8')
        return json.loads(s[s.index('={')+1:s.rfind(';')])
    c,q,i=(table(k) for k in ('composition','quality','induced'))
    typed=collections.defaultdict(set)
    m={'noun':'N','location':'N','adjective':'A','adjective2':'A','time':'T','adverb':'D','verb':'V'}
    for w in c['lexicon']:typed[w['text']].add(m[w['kind']])
    for w in q['words']:typed[w['text']].add({'N':'N','L':'N','V':'V','I':'I','T':'T','D':'D'}[w['pos']])
    obj={'source':'Project training segments + earlier assistant lexical hypotheses, not a certified dictionary','typed':{k:sorted(v) for k,v in typed.items()},'attested':{w['text']:w['count'] for w in i['lexicon']},'sourceIdentity':i['sourceIdentity']}
    raw=(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    actual=hashlib.sha256(raw).hexdigest()
    if actual!='16e5190617697f2023fbaca5a94252c4893fc15a08a5ce848f9b225fe55d8c54':
        raise ValueError('Different export: do not quietly mix research versions. SHA: '+actual)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);print(actual,dest)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('web',type=Path,help='Directory of already built/extracted Pages data assets');p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'data'/'lexicon.json');a=p.parse_args();build(a.web,a.out)
