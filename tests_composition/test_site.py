import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('composition_site',ROOT/'scripts/build_composition_site.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SiteTests(unittest.TestCase):
 def prepare(self,root):
  for name in m.REQUIRED:
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
  (root/'index.html').write_text('<html><div class="hero-copy">baseline</div></html>')
  self.manifest(root)
 def manifest(self,root):
  files=[]
  for p in sorted(root.rglob('*')):
   if p.is_file() and p.name!='build-manifest.json':
    raw=p.read_bytes();files.append({'path':p.relative_to(root).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  (root/'build-manifest.json').write_text(json.dumps({'schema':'verseloom.browser-demo.build.v1','files':files}))
 def test_entry_is_idempotent_and_manifest_matches(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.prepare(root);a=m.finish(root);b=m.finish(root)
   self.assertEqual(a,b);self.assertEqual((root/'index.html').read_text().count('id="composition-v2-entry"'),1)
   for e in json.loads((root/'build-manifest.json').read_text())['files']:
    self.assertEqual(hashlib.sha256((root/e['path']).read_bytes()).hexdigest(),e['sha256'])
 def test_changed_asset_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.prepare(root);(root/'composition-engine.js').write_text('tampered')
   with self.assertRaises(ValueError):m.finish(root)
 def test_missing_asset_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.prepare(root);(root/'composition-worker.js').unlink();self.manifest(root)
   with self.assertRaises(ValueError):m.finish(root)
 def test_unrecognized_output_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'build-manifest.json').write_text('{"schema":"other"}')
   with self.assertRaises(ValueError):m.finish(root)
if __name__=='__main__':unittest.main()
