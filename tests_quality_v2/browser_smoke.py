from playwright.sync_api import sync_playwright
import argparse,json,pathlib,shutil
ap=argparse.ArgumentParser(description="Render the self-contained workbench in Chromium memory, not a live-site test.")
ap.add_argument("--html",type=pathlib.Path,required=True)
ap.add_argument("--output",type=pathlib.Path,required=True)
ap.add_argument("--chromium",default=shutil.which("chromium"))
a=ap.parse_args()
if not a.chromium: raise SystemExit("Specify a Chromium executable with --chromium")
if a.output.exists(): raise SystemExit("Use a new output directory")
out=a.output;out.mkdir(parents=True)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=a.chromium,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 page=browser.new_page(viewport={'width':1440,'height':1100},accept_downloads=True)
 errors=[];requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:requests.append(r.url))
 page.set_content(a.html.read_text(encoding='utf-8'),wait_until='load')
 assert page.locator('#q-poem button').count()==0
 assert page.locator('#q-copy').is_disabled()
 page.locator('#q-auto').uncheck()
 samples=[]
 for length in ['5','7']:
  page.locator('#q-length').select_option(length)
  assert page.locator('#q-poem button').count()==0
  page.locator('#q-seed').fill('42')
  page.locator('#q-generate').click()
  page.wait_for_function("document.querySelector('#q-generate').disabled===false",timeout=90000)
  lines=page.locator('#q-poem button').all_text_contents()
  assert len(lines)==8, page.locator('#q-status').inner_text()
  assert all(len(l)==int(length)+1 for l in lines)
  page.locator('#q-poem button').nth(2).click()
  assert page.locator('#q-evidence').is_visible()
  assert page.locator('#q-readings details').count()==int(length)
  page.locator('#q-readings summary').first.click()
  with page.expect_download() as info:page.locator('#q-download').click()
  download=info.value;target=out/f'browser-{length}.json';download.save_as(target)
  result=json.loads(target.read_text());assert result['checks']['passed'] and result['provenance']['preselectedLines'] is False
  page.locator('#q-replay').click();page.wait_for_function("document.querySelector('#q-generate').disabled===false",timeout=90000)
  assert page.locator('#q-poem button').all_text_contents()==lines
  samples.append({'length':length,'lines':lines,'status':result['status'],'replay':True})
 page.screenshot(path=str(out/'desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(out/'mobile.png'),full_page=True)
 assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'horizontal overflow'
 page.locator('#q-generate').click();page.locator('#q-cancel').click()
 assert page.locator('#q-generate').is_enabled()
 assert '停止' in page.locator('#q-status').inner_text()
 assert not errors,errors
 assert all(u.startswith(('file:','data:','blob:')) for u in requests),requests
 report={'status':'PASS_IN_MEMORY_CHROMIUM_UI','browser':browser.version,'viewports':[[1440,1100],[390,844]],'samples':samples,'emptyStart':True,'wordEvidence':True,'jsonDownload':True,'cancel':True,'noHorizontalOverflow':True,'pageErrors':errors,'networkRequests':requests,'SafariTested':False,'liveSiteTested':False,'workerBrowserPathTested':False,'loadMethod':'Authored self-contained HTML rendered with set_content; no URL navigation'}
 (out/'browser-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 browser.close()
print('PASS_IN_MEMORY_CHROMIUM_UI')
