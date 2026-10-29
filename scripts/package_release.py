"""Create and verify a complete release archive before publishing its filename.

The output is create-only. A temporary archive is closed, CRC-checked and
verified against every manifest digest before it is atomically linked into
place. The final file is independently reopened and checked again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files(root: Path) -> list[Path]:
    excluded = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"release cannot silently follow a symlink: {relative}")
        if not path.is_file() or path.suffix in {".pyc", ".building"}:
            continue
        if relative.as_posix() == "ARTIFACT_MANIFEST.json":
            continue
        result.append(path)
    return result


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"archive CRC failed: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive names")
        manifests = [x for x in names if x.count("/") == 1
                     and x.endswith("/ARTIFACT_MANIFEST.json")]
        if len(manifests) != 1:
            raise ValueError("exactly one root manifest required")
        manifest_name = manifests[0]
        prefix = manifest_name.rsplit("/", 1)[0] + "/"
        manifest = json.loads(archive.read(manifest_name))
        expected = {manifest_name}
        for record in manifest["files"]:
            relative = record["path"]
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("unsafe archive path")
            name = prefix + relative
            if name in expected:
                raise ValueError("duplicate manifest path")
            expected.add(name)
            data = archive.read(name)
            if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
                raise ValueError(f"archive manifest mismatch: {name}")
        if set(names) != expected:
            raise ValueError("archive entries differ from complete manifest")
    return {"status": "passed", "entries": len(names),
            "manifest_files_verified": len(manifest["files"]),
            "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}


def build(root: Path, output: Path, version: str, report: str) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"release output is create-only: {output}")
    if root in output.parents:
        raise ValueError("release archive must be outside the project tree")
    if not (root / report).is_file():
        raise ValueError("a completed report must exist before packaging")
    files = release_files(root)
    manifest = {
        "project": "VerseMatrix Lab", "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "scope": "Context-to-span matrix matching development experiment and usable completion CLI; historical evidence retained.",
        "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                   "sha256": sha256(p.read_bytes())} for p in files],
    }
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    temporary_manifest = root / "ARTIFACT_MANIFEST.json.building"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".building", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files + [manifest_path]:
                archive.write(path, arcname=root.name + "/" + path.relative_to(root).as_posix())
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        before = verify_archive(temporary)
        # Atomic create, refusing to replace another finished release.
        os.link(temporary, output)
        temporary.unlink()
        after = verify_archive(output)
        if before != after:
            raise ValueError("final archive changed after atomic publication")
        return {"archive": str(output), "verification": after,
                "publication": "closed_verified_then_atomic_create"}
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", default="0.5.0")
    parser.add_argument("--report", default="docs/VerseMatrix_Progress.pdf")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify), ensure_ascii=False, indent=2))
    elif args.output:
        print(json.dumps(build(args.root, args.output, args.version, args.report), ensure_ascii=False, indent=2))
    else:
        parser.error("provide --output to create or --verify to verify")
