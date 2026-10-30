"""Build evidence: mobile styles must reach standalone files and be reproducible."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('site_builder',ROOT/'scripts/build_composition_site.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class MobilePackaging(unittest.TestCase):
    def test_inline_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('mobile-baseline.css','mobile-shared.css'):
                (root/name).write_text('body { min-width: 0; }')
            names=('index.html','standalone.html','composition.html','composition-offline.html','quality.html','quality-offline.html','induced.html')
            for name in names:(root/name).write_text('<html><head></head><body><div class="hero-copy"></div></body></html>')
            self.assertEqual(set(builder.mobile_styles(root)),set(names))
            first={name:(root/name).read_bytes() for name in names}
            builder.mobile_styles(root)
            self.assertEqual(first,{name:(root/name).read_bytes() for name in names})
            for name in names:self.assertEqual((root/name).read_text().count('id="verseloom-mobile-v1"'),1)
            self.assertIn('href="induced.html"',(root/'index.html').read_text())

    def test_partial_and_style_injection_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'mobile-baseline.css').write_text('body{}')
            with self.assertRaises(ValueError):builder.mobile_styles(root)
            (root/'mobile-shared.css').write_text('</style><script>bad</script>')
            with self.assertRaises(ValueError):builder.mobile_styles(root)
