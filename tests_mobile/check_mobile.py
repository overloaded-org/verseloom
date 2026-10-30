"""Render repository-built HTML in memory. Not real-device/live HTTP testing."""
import argparse, json, re, shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--chromium',default=shutil.which('chromium'));p.add_argument('--pages',default='');a=p.parse_args()
if a.output.exists():raise SystemExit('Refusing to overwrite test evidence')
a.output.mkdir(parents=True)
specs=[('index.html',['data.js','data7.js','engine.js','app.js'],'#form-select','#auto-seed','#seed-input','#generate-button','#poem-lines button','#replay-button'),('composition.html',['composition-data.js','composition-engine.js','composition-app.js'],'#length','#auto','#seed','#generate','#poem button','#replay'),('quality.html',['quality-data.js','quality-engine.js','quality-app.js'],'#q-length','#q-auto','#q-seed','#q-generate','#q-poem button','#q-replay'),('induced.html',['induced-meta.js','induced-app.js'],'#i-length','#i-auto','#i-seed','#i-generate','#i-poem button','#i-replay')]
checks=[];interactions=[];errors=[];requests=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path=a.chromium,headless=True,args=['--no-sandbox'])
 context=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=1,accept_downloads=True)
 for name,scripts,length_id,auto_id,seed_id,generate_id,poem_id,replay_id in specs:
  if a.pages and name not in a.pages.split(','):continue
  page=context.new_page();page.set_default_timeout(120000)
  page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda req:requests.append(req.url))
  text=(a.site/name).read_text()
  text=re.sub(r'<script\b[^>]*>.*?</script>','',text,flags=re.S|re.I)
  def inline_css(m):
   href=re.search(r'href=[\"\']([^\"\']+)',m[0]);return '<style>'+(a.site/href[1]).read_text()+'</style>' if href and href[1].endswith('.css') else ''
  text=re.sub(r'<link\b[^>]*>',inline_css,text,flags=re.I)
  # The base is used only to construct a Worker URL. No navigation/network occurs.
  text=text.replace('<head>','<head><base href="https://verseloom.test/">',1)
  page.set_content(text,wait_until='load')
  page.add_script_tag(content='globalThis.VERSELOOM_STANDALONE=true;globalThis.VERSELOOM_COMPOSITION_STANDALONE=true;globalThis.VERSELOOM_QUALITY_OFFLINE=true;')
  if name=='induced.html':
   body=(a.site/'induced-data.js').read_text()+'\n'+(a.site/'induced-engine.js').read_text()+'\n'+re.sub(r"importScripts\([^;]+;",'',(a.site/'induced-worker.js').read_text())
   page.evaluate('''body=>{const Native=Worker,url=URL.createObjectURL(new Blob([body],{type:'text/javascript'}));window.Worker=class {constructor(requested){if(!String(requested).endsWith('/induced-worker.js'))throw Error('Unexpected worker');return new Native(url);}};}''',body)
  for script in scripts:page.add_script_tag(content=(a.site/script).read_text())
  assert page.locator(poem_id).count()==0,(name,'prefilled')
  page.locator('details').evaluate_all('(xs)=>xs.forEach(x=>x.open=true)')
  page.locator(auto_id).uncheck()
  for length in ['5','7']:
   page.locator(length_id).select_option(length);page.locator(seed_id).fill('42');page.locator(generate_id).click()
   page.wait_for_function('(id)=>!document.querySelector(id).disabled',arg=generate_id,timeout=180000)
   lines=page.locator(poem_id).all_text_contents();assert len(lines)==8,(name,length,lines,page.locator('body').inner_text()[-1000:])
   page.locator(poem_id).nth(2).click()
   page.locator('details').evaluate_all('(xs)=>xs.forEach(x=>x.open=true)')
   for width in [320,375,390,430,768]:
    page.set_viewport_size({'width':width,'height':844})
    page.evaluate('() => new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
    metrics=page.evaluate('''() => ({width:innerWidth,scroll:document.documentElement.scrollWidth,controlSizes:[...document.querySelectorAll('input:not([type=checkbox]),select,button')].filter(e=>e.getBoundingClientRect().width>0).map(e=>({id:e.id,h:e.getBoundingClientRect().height,font:parseFloat(getComputedStyle(e).fontSize)}))})''')
    checks.append({'page':name,'length':int(length),**metrics})
    if metrics['scroll']>width+1:
     detail=page.evaluate('''width=>[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>width+1}).map(e=>({tag:e.tagName,id:e.id,c:e.className,w:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right,txt:e.textContent.slice(0,150)})).slice(-30)''',width)
     (a.output/'overflow.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
     page.screenshot(path=str(a.output/'overflow.png'),full_page=True)
    assert metrics['scroll']<=width+1,(name,length,width,metrics)
    if width<=430:
     assert all(c['h']>=43.9 for c in metrics['controlSizes']),(name,'small touch target',metrics)
    if width==390 and length=='7':page.screenshot(path=str(a.output/(name+'.png')),full_page=True)
   page.set_viewport_size({'width':390,'height':844})
   if name=='induced.html':
    page.locator(replay_id).click();page.wait_for_function('(id)=>!document.querySelector(id).disabled',arg=generate_id,timeout=180000)
    assert lines==page.locator(poem_id).all_text_contents()
   interactions.append({'page':name,'length':int(length),'lines':lines,'evidenceOpened':True,'replayed':name=='induced.html'})
   print(name,length,'PASS reflow',flush=True)
  page.close()
 assert not errors,errors
 assert all(url.startswith(('blob:','data:')) for url in requests),requests
 report={'status':'PASS_IN_MEMORY_CHROMIUM_MOBILE','browser':browser.version,'viewports':[320,375,390,430,768],'layoutChecks':len(checks),'checks':checks,'interactions':interactions,'pageErrors':errors,'requests':requests,'liveSiteTested':False,'iOSSafariTested':False,'workerMode':'induced uses real native Blob Worker with repository data/engine; other pages use same-source main-thread fallback','noNavigation':True}
 (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');browser.close()
