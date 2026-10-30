"""Verify trusted Pages artifact bytes before in-memory browser execution."""
import argparse, hashlib, io, json, re, tarfile, zipfile
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def extract(archive, run, artifact, repository, output):
    require(not output.exists(), 'Refusing to overwrite extracted assets')
    require(run['repository']['full_name'] == repository, 'Wrong repository')
    require(run['head_repository']['full_name'] == repository, 'Fork artifact forbidden')
    require(run['head_branch'] == 'main' and run['event'] in {'push', 'workflow_dispatch'}, 'Untrusted run event/ref')
    require(run['conclusion'] == 'success' and run['path'] == '.github/workflows/pages.yml', 'Not a successful Pages run')
    require(artifact['name'] == 'github-pages' and not artifact['expired'], 'Missing or expired Pages artifact')
    require(artifact['workflow_run']['id'] == run['id'] and artifact['workflow_run']['head_sha'] == run['head_sha'], 'Artifact run mismatch')
    digest = artifact.get('digest', '')
    require(re.fullmatch(r'sha256:[a-f0-9]{64}', digest), 'Expected GitHub artifact digest')
    require(archive.stat().st_size <= 64 * 1024 * 1024, 'Archive exceeds bound')
    raw = archive.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest.split(':')[1], 'Archive SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.namelist() == ['artifact.tar'], 'Unexpected archive layout')
        require(z.getinfo('artifact.tar').file_size <= 256 * 1024 * 1024, 'Expanded archive exceeds bound')
        tar = z.read('artifact.tar')
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        members = t.getmembers()
        require(len(members) <= 200, 'Too many archive entries')
        seen = set()
        for m in members:
            rel = Path(m.name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe archive path')
            require(m.isdir() or m.isfile(), 'Links and special files forbidden')
            if m.isdir():
                continue
            require(m.size <= 32 * 1024 * 1024 and rel.as_posix() not in seen, 'Invalid member size or duplicate')
            seen.add(rel.as_posix())
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(t.extractfile(m).read())
    manifest = json.loads((output / 'build-manifest.json').read_bytes())
    require(manifest['schema'] == 'verseloom.browser-demo.build.v1', 'Unknown manifest')
    expected = {'build-manifest.json'}
    for item in manifest['files']:
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() not in expected, 'Invalid manifest path')
        expected.add(rel.as_posix())
        body = (output / rel).read_bytes()
        require(len(body) == item['bytes'] and hashlib.sha256(body).hexdigest() == item['sha256'], 'Asset identity mismatch')
    require(seen == expected, 'Unlisted or absent assets')
    release = json.loads((output / 'release.json').read_bytes())
    require(release['commit'] == run['head_sha'], 'Release does not match source commit')
    return {'status': 'PASS_ARTIFACT_IDENTITY', 'sourceCommit': run['head_sha'], 'sourceRun': run['id'],
            'artifactId': artifact['id'], 'artifactDigest': digest, 'assetCount': len(manifest['files']),
            'liveURLTestedByThisStep': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('archive', 'run', 'artifact', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--repository', required=True)
    a = p.parse_args()
    print(json.dumps(extract(a.archive, json.loads(a.run.read_bytes()), json.loads(a.artifact.read_bytes()), a.repository, a.output), indent=2))


if __name__ == '__main__':
    main()
