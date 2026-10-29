"""Run risk tests and retain an actual machine-readable test receipt."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import numpy as np


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    target = root / "results"
    target.mkdir(parents=True, exist_ok=True)
    (target / "tests.log").write_text(buffer.getvalue(), encoding="utf-8")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for directory in [root / "src", root / "tests", root / "scripts"]
              for p in sorted(directory.rglob("*.py"))}
    receipt = {"status": "passed" if result.wasSuccessful() else "failed",
               "run_at_utc": datetime.now(timezone.utc).isoformat(),
               "tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "python": platform.python_version(), "numpy": np.__version__,
               "source_sha256": hashes, "interpretation": "engineering acceptance only; no poetry quality claim"}
    (target / "test_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:receipt[k] for k in ("status", "tests_run", "failures", "errors", "python", "numpy")}))
    sys.exit(0 if result.wasSuccessful() else 1)
