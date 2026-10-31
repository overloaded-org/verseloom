#!/usr/bin/env python3
"""Replay archived v0.1/v0.2 outputs before comparing a new diagnostic. No web writes."""
import argparse, collections, hashlib, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path

PIN = 'c2e64c5181bc5a1644dfe061dc63d270bc1095bde8bba0be9bc7a44585e128f0'
EXPECTED = '43695fda40f1de40dee471c035abcdaaf12aec1b0dc193ab6ffb66efe8fdd574'

def digest(b): return hashlib.sha256(b).hexdigest()

def run(source):
    if digest(source.read_bytes()) != PIN:
        raise ValueError('Not the pinned offline v0.2 evidence archive')
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None: raise ValueError('Corrupt source')
            archive.extractall(td)
        base = Path(td) / 'VerseLoom_v02'
        target = base / 'outputs/audit64.jsonl'
        reference = target.read_bytes()
        if digest(reference) != EXPECTED: raise ValueError('Wrong reference')
        target.unlink()
        env = {**os.environ, 'PYTHONHASHSEED':'0', 'PYTHONDONTWRITEBYTECODE':'1'}
        env.pop('PYTHONOPTIMIZE', None); env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, 'audit.py'], cwd=base, env=env, check=True, stdout=subprocess.DEVNULL)
        output = target.read_bytes()
        before = list(map(json.loads,reference.splitlines())); after = list(map(json.loads,output.splitlines()))
        if len(before) != 64 or len(after) != 64: raise ValueError('Missing or extra rows')
        equal = sum(a == b for a,b in zip(before, after))
        moves = collections.Counter(r['v01_label']+'>'+r['v02']['label'] for r in after if r['hard'])
        print(json.dumps({'replay_equal':equal,'n':64,'output_sha256':digest(output),'hard_transitions':moves},sort_keys=True))
        if equal != 64 or digest(output) != EXPECTED: raise ValueError('Baseline drift: do not publish comparisons')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('v02_zip',type=Path);run(p.parse_args().v02_zip)
