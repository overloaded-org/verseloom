#!/usr/bin/env python3
"""Bundle the quality workbench locally; no remote script/font dependencies."""
import argparse,base64,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(source,out):
 html=(source/'quality.html').read_text(encoding='utf-8')
 for name in ('quality-data.js','quality-engine.js','quality-app.js'):
  code=(source/name).read_text(encoding='utf-8')
  if name=='quality-app.js':code='globalThis.VERSELOOM_QUALITY_OFFLINE=true;\n'+code
  code=re.sub(r'</script',r'<\\/script',code,flags=re.I)
  html,count=re.subn(r'<script\s+src="'+re.escape(name)+r'"\s+defer></script>',lambda _: '<script>'+code+'</script>',html)
  if count!=1:raise ValueError('Unexpected script tag: '+name)
 icon=base64.b64encode((source/'favicon.svg').read_bytes()).decode()
 html=html.replace('href="favicon.svg"','href="data:image/svg+xml;base64,'+icon+'"')
 notice=(source/'quality-NOTICE.html').read_text(encoding='utf-8')
 body=re.search(r'<body[^>]*>(.*?)</body>',notice,re.S)
 if not body:raise ValueError('Missing notice body')
 html=html.replace('href="quality-NOTICE.html"','href="#offline-notice"')
 html=html.replace('</main>','<details id="offline-notice"><summary>数据来源与许可</summary>'+body.group(1)+'</details></main>')
 if re.search(r'<script[^>]+src=',html,re.I):raise ValueError('External script remains')
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html,encoding='utf-8')
 return {'offline':str(out),'bytes':out.stat().st_size}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'web/quality-offline.html');a=p.parse_args();print(json.dumps(build(ROOT/'web',a.output)))
