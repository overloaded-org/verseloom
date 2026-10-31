"""Real Pages URL, native Worker and DOM, not a physical iPhone. No auth or user data."""
import json, os, urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
base='https://overloaded-org.github.io/verseloom/'
assert json.load(urllib.request.urlopen(base+'release.json',timeout=25))['commit']==os.environ['SOURCE_SHA']
out=Path('output/live-endings');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
    browser=pw.chromium.launch();ctx=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,accept_downloads=True)
    page=ctx.new_page();errors=[];workers=[];page.on('pageerror',lambda e:errors.append(str(e)));page.on('worker',lambda w:workers.append(w.url));page.set_default_timeout(150000)
    page.goto(base+'induced.html');page.locator('a[href="endings.html"]').click();assert page.url==base+'endings.html';assert page.locator('#sheet').is_hidden()
    page.locator('#auto').uncheck();cases=[]
    for n,seed,expected in [(7,0,6695),(5,2,4021)]:
        page.locator('#length').select_option(str(n));page.locator('#seed').fill(str(seed));page.locator('#write').click();page.locator('#sheet').wait_for()
        prefix=page.locator('#prefix div').all_text_contents();assert len(prefix)==6
        for arm in ['original','greedy','joint','ordered']:
            page.locator(f'input[value="{arm}"]').check();assert prefix==page.locator('#prefix div').all_text_contents();assert len(page.locator('#ending div').all_text_contents())==2
        with page.expect_download() as dd: page.locator('#save').click()
        f=out/f'{n}-{seed}.json';dd.value.save_as(f);r=json.loads(f.read_bytes());assert r['legalPairs']==expected and r['prefix']==[s[:-1] for s in prefix] and r['lines']==r['displayedLines']
        assert [s[:-1] for s in page.locator('#ending div').all_text_contents()]==r['ordered']['ending']
        page.locator('#replay').click();page.locator('#sheet').wait_for();assert prefix==page.locator('#prefix div').all_text_contents()
        with page.expect_download() as dd: page.locator('#save').click()
        replay=out/f'{n}-replay.json';dd.value.save_as(replay);assert json.loads(replay.read_bytes())==r
        for width in [320,390,430]:
            page.set_viewport_size({'width':width,'height':844});assert page.evaluate('document.documentElement.scrollWidth')<=width+1
        page.screenshot(path=str(out/f'{n}-mobile.png'),full_page=True);cases.append({'length':n,'seed':seed,'legal':expected,'pairSha':r['legalPairSha256']})
    page.locator('#write').click();page.locator('#stop').click();assert page.locator('#sheet').is_hidden() and page.locator('#write').is_enabled();assert not errors,errors
    assert workers and all(w==base+'closure-worker.js' for w in workers),workers;version=browser.version;browser.close()
report={'status':'PASS_LIVE_CHROMIUM_MOBILE','url':base+'endings.html','sourceCommit':os.environ['SOURCE_SHA'],'browser':version,'cases':cases,'realWorker':True,'cancel':True,'deterministicExportReplay':True,'physicalSafari':False};print(json.dumps(report));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
