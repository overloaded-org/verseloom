"""Generate one unprompted statistical poem; no line or theme input is accepted."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.free_generation import run


def main():
    parser = argparse.ArgumentParser(description="训练语料片段自动组合八句五言；不接受人工填句。")
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/free_generation_v0.6.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="新建输出目录；已有目录一律拒绝覆盖")
    args = parser.parse_args()
    result = run(args.project_dir, args.output, args.config)
    print(json.dumps({"status": result["status"], "lines": result["lines"],
                      "method": result["method"], "output": str(args.output.resolve()),
                      "seconds": result["seconds"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
