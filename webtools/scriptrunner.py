"""
Run one of the repository's original scripts (``main.py``, ``generate_*.py``)
exactly as on the command line - but sandboxed, so it can live in the browser:

* a fresh temporary working directory (scripts write to the relative ``outputs/``),
* ``sys.argv`` set to the requested arguments,
* stdout captured (and, in Pyodide, also streamed live to the page),
* matplotlib rcParams, cwd, argv and NumPy's global RNG restored afterwards,
* every file the script produced collected for display / download.
"""

from __future__ import annotations

import io
import os
import runpy
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import numpy as np

from .core import Result, ToolError

APP_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = {".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
TEXT_TYPES = {".txt", ".csv", ".json", ".md"}


class _Tee(io.TextIOBase):
    """Write to the captured buffer and to the real stdout (live streaming in Pyodide)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:  # pragma: no cover - a closed stream must not kill the script
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # pragma: no cover
                pass


def run_script(script: str, argv: Optional[List[str]] = None, out: Optional[Result] = None,
               captions: Optional[dict] = None, max_log_chars: int = 60000) -> dict:
    """Execute ``script`` (path relative to the repo root) and add its outputs to ``out``."""
    import matplotlib

    path = APP_ROOT / script
    if not path.exists():
        raise ToolError(f"Skript nicht gefunden: {script}")
    captions = captions or {}
    old_cwd, old_argv, old_path = os.getcwd(), list(sys.argv), list(sys.path)
    old_rc = matplotlib.rcParams.copy()
    rng_state = np.random.get_state()
    buf = io.StringIO()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="trl-script-") as tmp:
        work = Path(tmp)
        (work / "outputs").mkdir()
        try:
            os.chdir(work)
            sys.argv = [script] + list(argv or [])
            if str(APP_ROOT) not in sys.path:
                sys.path.insert(0, str(APP_ROOT))
            if out is not None:
                out.progress(f"{script} läuft …")
            with redirect_stdout(_Tee(buf, sys.__stdout__ if sys.__stdout__ else io.StringIO())):
                try:
                    runpy.run_path(str(path), run_name="__main__")
                except SystemExit as exc:  # argparse / explicit exit
                    if exc.code not in (0, None):
                        raise ToolError(f"{script} wurde mit Code {exc.code} beendet.") from None
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            os.chdir(old_cwd)
            sys.argv = old_argv
            sys.path[:] = old_path
            matplotlib.rcParams.update(old_rc)
            np.random.set_state(rng_state)
        produced = sorted(p for p in work.rglob("*") if p.is_file())
        files = []
        for p in produced:
            rel = p.relative_to(work).as_posix()
            files.append({"path": rel, "bytes": p.stat().st_size})
            if out is None:
                continue
            suffix = p.suffix.lower()
            if suffix in IMAGE_TYPES:
                out.image(p.read_bytes(), captions.get(p.name, rel), mime=IMAGE_TYPES[suffix])
            elif suffix in TEXT_TYPES:
                out.download(p.name, p.read_text(encoding="utf-8", errors="replace"), "text/plain")
    seconds = time.perf_counter() - started
    log = buf.getvalue()
    if out is not None:
        out.metric("Laufzeit", seconds, "num", digits=1, unit="s")
        out.metric("Erzeugte Dateien", len(files), "int")
        out.metric("Ausgabezeilen", log.count("\n"), "int")
        if files:
            out.table("Erzeugte Dateien", ["Datei", "Größe (KB)"], [[f["path"], f["bytes"] / 1024] for f in files],
                      formats=["text", "num"])
    return {"seconds": seconds, "files": files, "log": log[-max_log_chars:]}
