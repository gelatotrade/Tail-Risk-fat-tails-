#!/usr/bin/env python3
"""
Assemble the GitHub Pages site into ``_site/``.

    site/                 static frontend (HTML, CSS, ES modules, market data)
    py/bundle.zip         src/, webtools/, main.py, generate_*.py  -> unpacked in Pyodide
    py/catalogue.json     tool catalogue, so the UI renders before Python boots
    gallery/*.png|gif     figures produced by the original scripts (outputs/)
    version.json          build id, commit, timestamp

``__BUILD__`` placeholders in HTML/JS are replaced by the build id so every
deploy busts browser caches exactly once.

Usage:  python scripts/build_site.py [--out _site]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_SOURCES = ["src", "webtools"]
PY_SCRIPTS = ["main.py"] + sorted(p.name for p in ROOT.glob("generate_*.py"))
IMPORT_RE = re.compile(r"""((?:from|import)\s*\(?\s*['"]\.{1,2}/[^'"?]+\.m?js)(['"])""")
TEXT_SUFFIXES = {".html", ".js", ".mjs", ".css", ".json", ".webmanifest", ".svg"}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "dev"


def write_bundle(dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    files = []
    for top in PY_SOURCES:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    files += [ROOT / name for name in PY_SCRIPTS]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            digest.update(rel.encode() + b"\0" + data)
            info = zipfile.ZipInfo(rel, date_time=(2020, 1, 1, 0, 0, 0))  # reproducible
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return digest.hexdigest()[:12]


def write_catalogue(dest: Path) -> dict:
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    from webtools import core

    catalogue = core.catalogue()
    catalogue["scripts"] = PY_SCRIPTS
    dest.write_text(json.dumps(catalogue, ensure_ascii=False, indent=1), encoding="utf-8")
    return catalogue


def copy_gallery(out: Path) -> int:
    manifest_path = ROOT / "site" / "data" / "gallery.json"
    if not manifest_path.exists():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = 0
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    for section in manifest["sections"]:
        for item in section["items"]:
            src = ROOT / "outputs" / item["file"]
            if not src.exists():
                raise FileNotFoundError(f"gallery.json references missing outputs/{item['file']}")
            shutil.copy2(src, out / "gallery" / item["file"])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "_site"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)

    bundle_hash = write_bundle(out / "py" / "bundle.zip")
    catalogue = write_catalogue(out / "py" / "catalogue.json")
    sha = git_sha()
    site_hash = hashlib.sha256()
    for path in sorted((ROOT / "site").rglob("*")):
        if path.is_file():
            site_hash.update(path.relative_to(ROOT).as_posix().encode() + path.read_bytes())
    build = f"{bundle_hash[:6]}{site_hash.hexdigest()[:6]}"

    for path in out.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if path.suffix in (".js", ".mjs"):
                # version relative ES-module imports too, so a deploy never mixes old and new modules
                text = IMPORT_RE.sub(lambda m: f"{m.group(1)}?v=__BUILD__{m.group(2)}", text)
            if "__BUILD__" in text:
                path.write_text(text.replace("__BUILD__", build), encoding="utf-8")

    n_gallery = copy_gallery(out)
    version = {"build": build, "commit": sha, "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "tools": len(catalogue["tools"]), "scripts": len(catalogue["scripts"])}
    (out / "version.json").write_text(json.dumps(version, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out}  build={build}  tools={version['tools']}  gallery={n_gallery}  size={size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
