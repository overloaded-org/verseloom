#!/usr/bin/env python3
"""Build one offline HTML with retained licence texts and no local-file links."""
import argparse, html, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def build(root):
    web=root/'web'; text=(web/'composition.html').read_text(encoding='utf-8')
    text=text.replace('<link rel="stylesheet" href="composition.css">','<style>'+ (web/'composition.css').read_text(encoding='utf-8')+'</style>')
    for name in ['composition-data.js','composition-engine.js','composition-app.js']:
        script=(web/name).read_text(encoding='utf-8')
        if name=='composition-app.js':script='globalThis.VERSELOOM_COMPOSITION_STANDALONE=true;\n'+script
        script=re.sub('</script',r'<\/script',script,flags=re.I)
        text=text.replace(f'<script src="{name}" defer></script>','<script>'+script+'</script>')
    notices=[]
    for name in ['licenses/poetry-MIT.txt','licenses/rhyme-NOTICE.md','licenses/tshet-uinh-CC0.txt']:
        notices.append('<pre>'+html.escape((web/name).read_text(encoding='utf-8'))+'</pre>')
    text=text.replace('<div id="license-text"></div>','<div id="license-text">'+''.join(notices)+'</div>')
    text=text.replace('href="index.html"','href="#"')
    text=text.replace('<a href="composition-offline.html" download>保存单文件离线版</a>', '<span>当前为单文件离线版</span>')
    text=text.replace('href="quality.html"','href="https://overloaded-org.github.io/verseloom/quality.html"')
    text=text.replace('<a href="NOTICE.html">原项目来源与许可说明</a>','<span>许可与来源随此文件保留。</span>')
    text=text.replace('href="licenses/tshet-uinh-CC0.txt"','href="#license-text"')
    if re.search(r'<script[^>]+src=',text):raise ValueError('External script dependency remains')
    return text
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite existing output')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(build(ROOT),encoding='utf-8');print(a.output)
