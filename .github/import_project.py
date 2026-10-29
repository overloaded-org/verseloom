"""One-time, hash-verified import of the prepared VerseLoom project."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

REPOSITORY = "overloaded-org/verseloom"
ZIP_SHA256 = "b8be2935fe859c8f5e3f829673363f96ad6b9a77d8ab52fbc9f4504f513acfc7"
ZIP_BYTES = 15125075
SOURCE_BYTES = 58966670
CI_PATH = ".github/workflows/ci.yml"
CI_STAGING_PATH = ".verseloom-import/ci-source.yml"
BOOTSTRAP_PATHS = {"README.md", ".github/import_project.py", ".github/workflows/import-project.yml"}
MANIFEST_PATH = "docs/GITHUB_UPLOAD_MANIFEST.json"
RECEIPT_PATH = "results/github_import_validation/github_actions_import_receipt.json"
TEST_LOG_PATH = "results/github_import_validation/github_actions_unit_tests.log"
AUDIT_PATH = "results/github_import_validation/github_actions_poem_audit.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git_blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def fetch_blob(sha):
    require(len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), "Invalid blob SHA")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/git/blobs/{sha}",
        headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(200001)
    require(len(raw) <= 200000, "Bootstrap blob too large")
    result = json.loads(raw)
    require(result["encoding"] == "base64" and result["sha"] == sha, "Blob response mismatch")
    content = base64.b64decode(result["content"], validate=False)
    require(git_blob_sha(content) == sha, "Bootstrap blob hash mismatch")
    return content


def safe_path(value):
    require(isinstance(value, str) and value, "Invalid manifest path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value, "Noncanonical manifest path")
    require(not any(part in {".", "..", ".git"} for part in path.parts), "Unsafe manifest path")
    require("\\" not in value and not any(ord(c) < 32 for c in value), "Unsafe path characters")
    return path


def verify_bytes(content, entry):
    require(len(content) == entry["bytes"], "Source size mismatch: " + entry["path"])
    require(hashlib.sha256(content).hexdigest() == entry["sha256"], "Source SHA256 mismatch: " + entry["path"])
    require(git_blob_sha(content) == entry["git_blob_sha"], "Source Git blob mismatch: " + entry["path"])


def verify_tree(root, entries):
    for entry in entries:
        path = root / entry["path"]
        require(path.is_file() and not path.is_symlink(), "Source is not a regular file")
        verify_bytes(path.read_bytes(), entry)
        expected_mode = 0o755 if entry["mode"] == "100755" else 0o644
        require(stat.S_IMODE(path.stat().st_mode) == expected_mode, "Source mode mismatch")


def extract_verified(archive, destination):
    require(archive.stat().st_size == ZIP_BYTES, "Archive size mismatch")
    require(hashlib.sha256(archive.read_bytes()).hexdigest() == ZIP_SHA256, "Archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        require(len(infos) == 365 and len({item.filename for item in infos}) == 365, "ZIP member count mismatch")
        require(all(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16) for item in infos), "Unsafe ZIP member")
        require(sum(item.file_size for item in infos) < 70 * 1024 * 1024, "Expanded archive is too large")
        manifest_bytes = bundle.read("UPLOAD_MANIFEST.json")
        manifest = json.loads(manifest_bytes)
        entries = manifest["files"]
        require(len(entries) == 364, "Manifest file count mismatch")
        paths = [str(safe_path(entry["path"])) for entry in entries]
        require(len(set(paths)) == len(paths), "Duplicate manifest path")
        require(sum(entry["bytes"] for entry in entries) == SOURCE_BYTES, "Source byte total mismatch")
        require({item.filename for item in infos} == {"UPLOAD_MANIFEST.json"} | {"versematrix-lab/" + path for path in paths}, "Unexpected ZIP members")
        require(not ({MANIFEST_PATH, RECEIPT_PATH, TEST_LOG_PATH, AUDIT_PATH, CI_STAGING_PATH} & set(paths)), "Import metadata path collision")
        require(set(path for path in paths if path.startswith(".github/workflows/")) == {CI_PATH}, "Unexpected source workflow")
        require(".github/import_project.py" not in paths, "Bootstrap path collision")
        for entry in entries:
            require(entry["mode"] in {"100644", "100755"}, "Invalid source file mode")
            require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid source byte count")
            content = bundle.read("versematrix-lab/" + entry["path"])
            verify_bytes(content, entry)
            path = destination / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(0o755 if entry["mode"] == "100755" else 0o644)
    verify_tree(destination, entries)
    return entries, manifest_bytes


def clean_environment():
    environment = dict(os.environ)
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "TRANSFER_BLOB_SHA", "TRANSFER_HELPER_SHA", "HOSTED_APPS_HELPER_SHA"):
        environment.pop(key, None)
    return environment


def validate_project(source, temporary):
    environment = clean_environment()
    environment.update(PYTHONPATH=str(source / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    receipt = temporary / "unit-test-receipt.json"
    log = temporary / "unit-tests.log"
    harness = """import json,sys,unittest
suite=unittest.defaultTestLoader.discover('tests')
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'unexpected_successes':len(result.unexpectedSuccesses)}
with open(sys.argv[1],'w',encoding='utf-8') as handle: json.dump(receipt,handle,sort_keys=True)
sys.exit(0 if result.wasSuccessful() and result.testsRun==175 and not result.skipped else 1)
"""
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-c", harness, str(receipt)], cwd=source, env=environment,
                       stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=180)
    test_result = json.loads(receipt.read_text())
    require(test_result == {"tests_run":175,"failures":0,"errors":0,"skipped":0,"unexpected_successes":0}, "Unexpected test results")
    audit_path = temporary / "poem-audit.json"
    audit_run = subprocess.run([sys.executable, "scripts/audit_free_poem_v06.py", "--project-dir", str(source),
                               "--run", str(source / "results/free_generation_v06_run001"), "--output", str(audit_path)],
                              cwd=source, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=240)
    require(audit_run.returncode == 0, "Frozen poem independent audit failed")
    audit = json.loads(audit_path.read_text())
    require(audit["status"] == "PASS_INDEPENDENT_SOURCE_AND_SELECTED_PATH_AUDIT", "Unexpected audit status")
    require(audit["formal_generation_executed_by_auditor"] is False and audit["real_output_altered"] is False, "Audit altered output")
    return test_result, log, audit_path


def index_entries(root):
    output = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    entries = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, sha, stage = metadata.decode("ascii").split()
        require(stage == "0", "Unmerged index")
        entries[path.decode("utf-8")] = (mode, sha)
    return entries


def write_file(path, content, mode="100644"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755 if mode == "100755" else 0o644)


def main():
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "Wrong repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/main", "Wrong branch")
    root = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    bootstrap_sha = os.environ["GITHUB_SHA"]
    require(git(root, "rev-parse", "HEAD") == bootstrap_sha, "Checkout is not the triggering commit")
    original_index = index_entries(root)
    require(set(original_index) == BOOTSTRAP_PATHS and not git(root, "status", "--porcelain"), "Unexpected bootstrap tree")
    with tempfile.TemporaryDirectory(prefix="verseloom-import-", dir=os.environ["RUNNER_TEMP"]) as temporary_name:
        temporary = Path(temporary_name)
        helpers = temporary / "helpers"
        helpers.mkdir(mode=0o700)
        for name, key in (("library_file_transfer.py", "TRANSFER_HELPER_SHA"), ("library_hosted_apps.py", "HOSTED_APPS_HELPER_SHA")):
            write_file(helpers / name, fetch_blob(os.environ[key]))
            (helpers / name).chmod(0o600)
        transfer_bytes = fetch_blob(os.environ["TRANSFER_BLOB_SHA"])
        transfer = json.loads(transfer_bytes)
        require(transfer["library_file_id"] == "libfile_b0160d73a4f88191b8cc8772d1494819" and transfer["file_id"] == "file_00000000db9081f5910747c5743ccc0d", "Wrong transfer identity")
        require(transfer["size_bytes"] == ZIP_BYTES and transfer["current_version_number"] == 0, "Wrong transfer version")
        require(transfer.get("workspace_path") is None and transfer["headers"] == {} and transfer["method"] == "GET", "Unexpected transfer authorization")
        address = urllib.parse.urlsplit(transfer["download_url"])
        query = urllib.parse.parse_qs(address.query)
        require(address.scheme == "https" and address.hostname.endswith(".oaiusercontent.com") and query.get("sp") == ["r"] and query.get("sr") == ["b"], "Transfer is not scoped to a read-only file")
        archive = temporary / "VerseMatrix_GitHub_Ready.zip"
        downloaded = subprocess.run([sys.executable, str(helpers / "library_file_transfer.py"), "materialize", str(archive)],
                                    input=transfer_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env=clean_environment(), timeout=240)
        require(downloaded.returncode == 0, "Official transfer helper failed; private output suppressed")
        print("Prepared archive downloaded; validating all bytes.", flush=True)
        source = temporary / "source"
        source.mkdir()
        entries, manifest_bytes = extract_verified(archive, source)
        print("Verified 364 source files; running 175 tests and frozen poem audit.", flush=True)
        test_result, test_log, audit_path = validate_project(source, temporary)
        verify_tree(source, entries)
        run_url = f"https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        receipt = {
            "status": "VERIFIED_FOR_SOURCE_IMPORT", "archive_sha256": ZIP_SHA256, "archive_bytes": ZIP_BYTES,
            "source_files": len(entries), "source_bytes": SOURCE_BYTES, "source_manifest": MANIFEST_PATH,
            "bootstrap_commit": bootstrap_sha, "workflow_run_url": run_url, "unit_tests": test_result,
            "all_source_hashes_rechecked_after_tests": True, "poem_generation_executed": False,
            "ci_staging_path": CI_STAGING_PATH, "ci_final_path": CI_PATH,
            "note": "The preserved upload manifest describes the prepared archive. Final README branding is a later explicit edit."
        }
        extra = {
            MANIFEST_PATH: manifest_bytes,
            RECEIPT_PATH: (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            TEST_LOG_PATH: test_log.read_bytes(), AUDIT_PATH: audit_path.read_bytes(),
        }
        expected = {path: value for path, value in original_index.items() if path != "README.md"}
        paths_to_add = []
        for entry in entries:
            destination_path = CI_STAGING_PATH if entry["path"] == CI_PATH else entry["path"]
            write_file(root / destination_path, (source / entry["path"]).read_bytes(), entry["mode"])
            expected[destination_path] = (entry["mode"], entry["git_blob_sha"])
            paths_to_add.append(destination_path)
        for path, content in extra.items():
            write_file(root / path, content)
            expected[path] = ("100644", git_blob_sha(content))
            paths_to_add.append(path)
        git(root, "config", "core.autocrlf", "false")
        git(root, "config", "core.filemode", "true")
        git(root, "add", "--", *paths_to_add)
        require(index_entries(root) == expected, "Git index hashes, modes, or paths differ from verified import")
        remote = git(root, "ls-remote", "origin", "refs/heads/main").split()
        require(len(remote) == 2 and remote[0] == bootstrap_sha, "Remote advanced during import; stopping without force")
        git(root, "config", "user.name", "github-actions[bot]")
        git(root, "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        git(root, "commit", "-m", "Import verified VerseLoom source, frozen experiments, and autonomous poem")
        git(root, "push", "origin", "HEAD:refs/heads/main")
        print(json.dumps({"status":"SOURCE_IMPORTED", "commit_sha":git(root,"rev-parse","HEAD"), "tests":test_result, "run_url":run_url}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
