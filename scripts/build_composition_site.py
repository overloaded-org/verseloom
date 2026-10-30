#!/usr/bin/env python3
"""Finish an existing Pages build with a v2 entry and a same-build manifest."""
import argparse, hashlib, json, os, re
from pathlib import Path

REQUIRED = ('composition.html', 'composition.css', 'composition-data.js',
            'composition-engine.js', 'composition-app.js', 'composition-worker.js',
            'composition-offline.html', 'licenses/tshet-uinh-CC0.txt')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def finish(site):
    site = site.resolve()
    marker = site / 'build-manifest.json'
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'verseloom.browser-demo.build.v1':
        raise ValueError('Unrecognized site directory')
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe manifest path')
        raw = (site / rel).read_bytes()
        if len(raw) != item['bytes'] or digest(raw) != item['sha256']:
            raise ValueError('Site changed since packaging: ' + item['path'])
    for name in REQUIRED:
        if not (site / name).is_file():
            raise ValueError('Missing composition asset: ' + name)
    index = site / 'index.html'
    text = index.read_text(encoding='utf-8')
    entry = '<p id="composition-v2-entry"><a class="text-link" href="composition.html">开始体验：词组共构 v2 · 五言／七言 →</a></p>'
    if 'id="composition-v2-entry"' not in text:
        anchor = '<div class="hero-copy">'
        if text.count(anchor) != 1:
            raise ValueError('Homepage entry anchor changed')
        text = text.replace(anchor, anchor + '\n        ' + entry, 1)
        index.write_text(text, encoding='utf-8')
    sha = os.environ.get('GITHUB_SHA', '')
    if sha and not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('Invalid build commit')
    release = {'schema':'verseloom.pages-release.v1', 'commit':sha or None,
               'entry':'composition.html', 'engineVersion':'verseloom.phrase-composition.v2'}
    (site/'release.json').write_text(json.dumps(release, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    files = []
    for p in sorted(site.rglob('*')):
        if p.is_symlink():
            raise ValueError('Site contains symbolic links')
        if not p.is_file() or p == marker:
            continue
        raw = p.read_bytes()
        files.append({'path':p.relative_to(site).as_posix(), 'bytes':len(raw), 'sha256':digest(raw)})
    manifest['files'] = files
    marker.write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    return {'status':'composition_site_ready', 'entry':'composition.html',
            'files':len(files), 'manifestSha256':digest(marker.read_bytes()), 'commit':release['commit']}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path('_site'))
    print(json.dumps(finish(parser.parse_args().site), ensure_ascii=False))
