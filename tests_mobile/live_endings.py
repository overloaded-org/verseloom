"""Check actual Pages + Worker. Intentionally hold the real app script for the readiness test."""
import json, os, urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
base='https://overloaded-org.github.io/verseloom/'
release=json.load(urllib.request.urlopen(base+'release.json',timeout=25))['commit'];assert release==os.environ['SOURCE_SHA'],(release,os.environ['SOURCE_SHA'])
out=Path('output/live-endings');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
    browser=pw.chromium.launch();ctx=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,accept_downloads=True)
    page=ctx.new_page();errors=[];workers=[];events=[];cases=[];held=[];mobile_widths=[320,390,430]
    page.on('pageerror',lambda e:errors.append(str(e)));page.on('worker',lambda w:workers.append(w.url));page.on('requestfailed',lambda r:events.append([r.url,r.failure]));page.on('console',lambda m:events.append([m.type,m.text]) if m.type=='error' else None);page.set_default_timeout(120000)
    def complete():
        page.wait_for_function("!document.getElementById('write').disabled");assert page.locator('#sheet').is_visible(),(page.locator('#status').inner_text(),errors,workers,events)
    try:
        # No mock script or fake result. Hold only transport, then release the actual response.
        gate='**/closure-app.js';page.route(gate,lambda route:held.append(route));page.goto(base+'endings.html',wait_until='commit');page.locator('#write').wait_for();
        assert page.locator('#write').is_disabled() and page.locator('#sheet').is_hidden() and page.evaluate("document.getElementById('form').onsubmit===null"),'Ungated startup'
        for _ in range(200):
            if held:break
            page.wait_for_timeout(25)
        assert len(held)==1;held.pop().continue_();page.wait_for_function("!document.getElementById('write').disabled && typeof document.getElementById('form').onsubmit==='function'");page.unroute(gate)
        page.goto(base+'induced.html');page.locator('a[href="endings.html"]').click();page.wait_for_url(base+'endings.html');assert page.locator('#sheet').is_hidden();page.locator('#auto').uncheck()
        for n,seed,expected in [(7,0,6695),(5,2,4021)]:
            page.locator('#length').select_option(str(n));page.locator('#seed').fill(str(seed));page.locator('#write').click();complete()
            prefix=page.locator('#prefix div').all_text_contents();assert len(prefix)==6
            for arm in ['original','greedy','joint','ordered']:
                page.locator(f'input[value="{arm}"]').check();assert prefix==page.locator('#prefix div').all_text_contents();assert len(page.locator('#ending div').all_text_contents())==2
            with page.expect_download() as dd: page.locator('#save').click()
            f=out/f'{n}-{seed}.json';dd.value.save_as(f);r=json.loads(f.read_bytes());assert r['legalPairs']==expected and r['prefix']==[s[:-1] for s in prefix] and r['lines']==r['displayedLines']
            assert [s[:-1] for s in page.locator('#ending div').all_text_contents()]==r['ordered']['ending']
            page.locator('#replay').click();complete();assert prefix==page.locator('#prefix div').all_text_contents()
            with page.expect_download() as dd: page.locator('#save').click()
            replay=out/f'{n}-replay.json';dd.value.save_as(replay);assert json.loads(replay.read_bytes())==r
            for width in mobile_widths:
                page.set_viewport_size({'width':width,'height':844});assert page.evaluate('document.documentElement.scrollWidth')<=width+1
            page.screenshot(path=str(out/f'{n}-mobile.png'),full_page=True);cases.append({'length':n,'seed':seed,'legal':expected,'pairSha':r['legalPairSha256']})
        page.locator('#write').click();page.locator('#stop').click();assert page.locator('#sheet').is_hidden() and page.locator('#write').is_enabled();assert not errors,errors
        assert workers and all(w==base+'closure-worker.js' for w in workers),workers
        report={'status':'PASS_LIVE_CHROMIUM_MOBILE','url':base+'endings.html','sourceCommit':release,'browser':browser.version,'cases':cases,'widths':mobile_widths,'delayedScriptGate':True,'realWorker':True,'cancel':True,'deterministicExportReplay':True,'physicalSafari':False};print(json.dumps(report));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        (out/'diagnostics.json').write_text(json.dumps({'release':release,'errors':errors,'events':events,'workers':workers,'status':page.locator('#status').text_content(timeout=2000) if page.locator('#status').count() else None},ensure_ascii=False,indent=2));page.screenshot(path=str(out/'surface.png'));browser.close()
