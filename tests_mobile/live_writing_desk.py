"""Exercise deployed Pages bytes, including native Workers, on mobile browser engines."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE='https://overloaded-org.github.io/verseloom/'
SHA=os.environ['SOURCE_SHA']; NAME=os.environ['BROWSER']; assert NAME in ('chromium','webkit')
OUT=Path('output/live-desk'); OUT.mkdir(parents=True,exist_ok=True)

def get(name):
    req=urllib.request.Request(BASE+name+'?v='+SHA,headers={'Cache-Control':'no-cache'})
    with urllib.request.urlopen(req,timeout=30) as resp: return resp.read()

def provenance():
    assert json.loads(get('release.json'))['commit']==SHA,'Wrong live release'
    css=get('writing-desk.css');assert css==Path('web/writing-desk.css').read_bytes(),'Wrong live stylesheet'
    return hashlib.sha256(css).hexdigest()

def layout(page,width,first=False):
    page.set_viewport_size({'width':width,'height':568 if width==320 else 844})
    page.evaluate('window.scrollTo(0,0)')
    size=page.evaluate('''() => {const r=document.querySelector('#i-generate').getBoundingClientRect();return {overflow:document.documentElement.scrollWidth-innerWidth,buttonBottom:r.bottom,height:innerHeight,buttonHeight:r.height}}''')
    assert size['overflow']<=1,(width,size)
    assert size['buttonHeight']>=44,(width,size)
    if first: assert 0<size['buttonBottom']<=size['height'],(width,size)
    return {**size,'width':width}

def wait_poem(page,n):
    page.wait_for_function("!document.getElementById('i-generate').disabled && document.getElementById('i-sheet').getAttribute('aria-busy')==='false'")
    lines=page.locator('#i-poem .poem-line').all_text_contents()
    assert len(lines)==8 and all(len(l[:-1])==n for l in lines),(n,lines,page.locator('#i-status').inner_text())
    return lines

def record(page,tag):
    with page.expect_download() as dl:page.locator('#i-export').tap()
    f=OUT/(tag+'.json');dl.value.save_as(f);return json.loads(f.read_bytes())

with sync_playwright() as pw:
    digest=provenance();br=getattr(pw,NAME).launch();context=br.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=1,accept_downloads=True)
    page=context.new_page();page.set_default_timeout(150000);errors=[];workers=[];checks=[];cases=[]
    page.on('worker',lambda w:workers.append(w.url));page.on('pageerror',lambda e:errors.append(str(e)))
    try:
        page.goto(BASE,wait_until='domcontentloaded');page.locator('#induced-v1-entry a').tap();page.wait_for_url(BASE+'induced.html');page.wait_for_function("!document.getElementById('i-generate').disabled")
        assert page.locator('[data-edition="mobile-writing-desk-v1"]').count()==1
        assert not page.locator('#i-poem button').count() and not page.locator('#i-settings').evaluate('(e)=>e.open')
        for width in (320,390,430,768):checks.append(layout(page,width,True))
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(OUT/'first-screen.png'))
        for n in (5,7):
            page.locator('#i-length').select_option(str(n));assert not page.locator('#i-poem button').count();page.locator('#i-settings').evaluate('(e)=>e.open=true');page.locator('#i-auto').uncheck();page.locator('#i-seed').fill('42');page.locator('#i-settings').evaluate('(e)=>e.open=false');page.locator('#i-generate').tap()
            lines=wait_poem(page,n);r=record(page,f'{n}-manual');assert r['options']['seed']==42 and r['options']['mode']=='relations' and r['lines']==[l[:-1] for l in lines] and all(c['passed'] for c in r['checks']);
            for width in (320,390,430):checks.append(layout(page,width));
            page.locator('#i-poem button').nth(2).tap();assert page.locator('#i-inspection').is_visible();assert r['lines'][2] in page.locator('#i-detail').text_content();assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
            page.locator('#i-replay').tap();assert wait_poem(page,n)==lines
            page.set_viewport_size({'width':390,'height':844});page.locator('#i-sheet').evaluate("(e)=>e.scrollIntoView({block:'start'})");page.screenshot(path=str(OUT/f'poem-{n}.png'))
            cases.append({'length':n,'seed':42,'count':len(lines),'replay':True})
        page.locator('#i-settings').evaluate('(e)=>e.open=true');page.locator('#i-auto').check();page.locator('#i-settings').evaluate('(e)=>e.open=false');page.locator('#i-again').tap();wait_poem(page,7);again=record(page,'seven-again');assert again['options']['seed']!=42
        page.locator('#i-generate').tap();page.locator('#i-cancel').tap();assert page.locator('#i-generate').is_enabled() and '停止' in page.locator('#i-status').inner_text()
        page.locator('a[href="endings.html"]').tap();page.wait_for_url(BASE+'endings.html');assert not page.locator('#sheet').is_visible();assert page.locator('#write').is_enabled();assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
        assert any(w.endswith('/induced-worker.js') for w in workers),workers
        assert not errors,errors;assert provenance()==digest
        (OUT/'report.json').write_text(json.dumps(dict(status='PASS_LIVE_MOBILE_WRITING_DESK',sha=SHA,engine=NAME,version=br.version,cssSha256=digest,cases=cases,layouts=checks,repeatFromPoem=True,cancel=True,workers=workers,errors=errors,physicalDevice=False),ensure_ascii=False,indent=2))
        print('PASS_LIVE_MOBILE_WRITING_DESK',SHA,NAME,br.version)
    finally:
        try:page.screenshot(path=str(OUT/'last-screen.png'),timeout=5000)
        finally:br.close()
