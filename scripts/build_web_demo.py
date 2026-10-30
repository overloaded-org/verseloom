#!/usr/bin/env python3
"""Package the browser demo and a self-contained offline HTML, without a CDN."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
MARKER = "build-manifest.json"


def replace_once(pattern: str, replacement: str, text: str) -> str:
    result, count = re.subn(pattern, lambda _: replacement, text, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError(f"Expected exactly one matching asset tag; found {count}: {pattern}")
    return result


def script_text(text: str) -> str:
    return re.sub(r"</script", r"<\\/script", text, flags=re.IGNORECASE)


def standalone(source: Path) -> str:
    html = (source / "index.html").read_text(encoding="utf-8")
    css = (source / "styles.css").read_text(encoding="utf-8")
    if re.search(r"</style", css, flags=re.IGNORECASE):
        raise ValueError("Unexpected style closing tag in stylesheet")
    html = replace_once(r'<link\b[^>]*href=["\'](?:\./)?styles\.css["\'][^>]*>', "<style>\n" + css + "\n</style>", html)
    for filename in ("data.js", "engine.js", "app.js"):
        text = (source / filename).read_text(encoding="utf-8")
        if filename == "app.js":
            text = ("window.VERSELOOM_STANDALONE = true;\n"
                    "const startVerseLoomOffline = () => {\n" + text + "\n};\n"
                    "if (document.readyState === 'loading') {\n"
                    "document.addEventListener('DOMContentLoaded', startVerseLoomOffline, {once:true});\n"
                    "} else { startVerseLoomOffline(); }\n")
        html = replace_once(r'<script\b[^>]*src=["\'](?:\./)?' + re.escape(filename) + r'["\'][^>]*>\s*</script>',
                            "<script>\n" + script_text(text) + "\n</script>", html)
    favicon = source / "favicon.svg"
    if favicon.exists():
        encoded = base64.b64encode(favicon.read_bytes()).decode("ascii")
        html = re.sub(r'(?P<quote>["\'])(?:\./)?favicon\.svg(?P=quote)',
                      lambda _: '"data:image/svg+xml;base64,' + encoded + '"', html)
    if re.search(r'<script\b[^>]*src=', html, flags=re.IGNORECASE):
        raise ValueError("Offline HTML still has an external script dependency")
    notice_html = (source / "NOTICE.html").read_text(encoding="utf-8")
    notice_match = re.search(r"<main[^>]*>(.*?)</main>", notice_html, flags=re.DOTALL)
    if not notice_match:
        raise ValueError("Missing portable source and license notice")
    notice = notice_match.group(1).replace('href="./index.html"', 'href="#"')
    html = re.sub(r'href=["\'](?:\./)?NOTICE\.html["\']', 'href="#offline-notice"', html)
    details = ('<details id="offline-notice" class="offline-source-note"><summary>数据来源与许可</summary>'
               + notice + '</details>')
    html = replace_once(r"</body>", details + "\n</body>", html)
    portable_style = ('<style>.offline-source-note{max-width:1080px;margin:24px auto;padding:24px;line-height:1.85}'
                      '.offline-source-note pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}'
                      '.offline-source-note h1{font-size:24px}.offline-source-note h2{font-size:20px}</style>')
    html = replace_once(r"</head>", portable_style + "\n</head>", html)
    return html


def build(output: Path) -> dict:
    source = ROOT / "web"
    output = output.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Build output must be separate from the source tree")
    required = ("index.html", "styles.css", "data.js", "engine.js", "app.js", "worker.js", "NOTICE.md", "NOTICE.html")
    for name in required:
        if not (source / name).is_file():
            raise ValueError("Missing browser asset: " + name)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Site source must not contain symbolic links")
    if any(part.startswith(".") for path in files for part in path.relative_to(source).parts):
        raise ValueError("Site source must not contain hidden files")
    output.mkdir(parents=True, exist_ok=True)
    old_files = [p for p in output.rglob("*") if p.is_file()]
    if old_files:
        marker = output / MARKER
        if not marker.is_file():
            raise ValueError("Refusing to overwrite an unrecognized output directory")
        old_manifest = json.loads(marker.read_text(encoding="utf-8"))
        owned = {item["path"] for item in old_manifest["files"]} | {MARKER}
        if any(str(path.relative_to(output)) not in owned for path in old_files):
            raise ValueError("Unexpected file in build output; no files were removed")
        incoming = {str(p.relative_to(source)) for p in files} | {"standalone.html", MARKER}
        for path in old_files:
            if str(path.relative_to(output)) not in incoming:
                path.unlink()
    for path in files:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output / "standalone.html").write_text(standalone(source), encoding="utf-8")
    manifest = {"schema": "verseloom.browser-demo.build.v1", "files": []}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != MARKER):
        content = path.read_bytes()
        manifest["files"].append({"path": str(path.relative_to(output)), "bytes": len(content),
                                  "sha256": hashlib.sha256(content).hexdigest()})
    (output / MARKER).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "built", "directory": str(output), "files": len(manifest["files"]),
            "offline_html": str(output / "standalone.html")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
