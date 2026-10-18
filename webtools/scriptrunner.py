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
CAPTIONS = {
    "fat_tails_comparison.png": "Fette Ränder vs. Normalverteilung",
    "physics_finance_mapping.png": "Physik → Finanzen",
    "tail_index_interpretation.png": "Tail-Index α richtig lesen",
    "var_comparison.png": "VaR-Methoden im Vergleich",
    "early_warning_signals.png": "Kritische Verlangsamung",
    "tail_risk_surface_3d.png": "3D-Tail-Risk-Fläche",
    "tail_risk_surface_animation.gif": "Animierte Risikofläche",
    "tail_risk_3d_dashboard.png": "3D-Phasenraum-Dashboard",
    "early_warning_3d_phase_space.png": "Frühwarnung im 3D-Phasenraum",
    "warning_matrix.png": "Warnmatrix",
    "early_warning_indicators.png": "Die fünf Frühwarnindikatoren",
    "early_warning_dashboard_demo.png": "Frühwarn-Dashboard",
    "crash_timeline_indicators.png": "Zeitachse der Signale",
    "crash_comparison_indicators.png": "Indikatoren im Crash-Vergleich",
    "ews_analysis_jan2026.png": "Frühwarnanalyse Januar 2026",
    "crisis_comparison.png": "Krisenvergleich",
    "crash_comparison_with_dates.png": "Crashs mit Datumsachsen",
    "volmageddon_2018_deep_dive.png": "Volmageddon 2018 im Detail",
    "covid_2020_deep_dive.png": "COVID-Crash 2020 im Detail",
    "bear_2022_deep_dive.png": "Bärenmarkt 2022 im Detail",
    "tariff_2025_deep_dive.png": "Zoll-Crash 2025 im Detail",
    "covid_crash_analysis.png": "COVID-Crash: Analyse",
    "bear_2022_analysis.png": "Bärenmarkt 2022: Analyse",
    "tariff_crash_analysis.png": "Zoll-Crash 2025: Analyse",
    "tariff_crash_deep_dive.png": "Zoll-Crash 2025: Vertiefung",
    "covid_crash_dashboard.png": "Dashboard COVID-19",
    "bear_2022_dashboard.png": "Dashboard Bärenmarkt 2022",
    "tariff_crash_dashboard.png": "Dashboard Zoll-Crash 2025",
    "crisis_dashboard.png": "Krisen-Dashboard",
    "full_dashboard.png": "Vollständiges Dashboard",
    "current_market_dashboard.png": "Aktuelles Markt-Dashboard",
    "current_market_analysis.png": "Aktuelle Marktanalyse",
    "current_state_jan2026.png": "Marktlage Januar 2026",
    "crisis_vs_current_comparison.png": "Krisen vs. heute",
    "analysis_dashboard.png": "Tail-Risk-Dashboard deiner Daten",
    "tail_risk_report.txt": "Bericht",
    "analysis_report.txt": "Bericht",
}
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


QUALITY = {
    "preview": 90,     # fast in the browser, small payload
    "standard": 150,   # what most scripts use
    "original": None,  # whatever the script asks for (up to 300 dpi)
}
QUALITY_PARAM = {
    "name": "quality", "type": "select", "default": "preview", "label": "Bildqualität",
    "options": [{"value": "preview", "label": "Vorschau (90 dpi, schnell)"},
                {"value": "standard", "label": "Standard (150 dpi)"},
                {"value": "original", "label": "Original (wie im Skript, bis 300 dpi)"}],
    "help": "Die Skripte speichern mit bis zu 300 dpi. Im Browser kostet das viel Rechenzeit – für die Analyse reicht die Vorschau.",
}


class _DpiCap:
    """Clamp the dpi of every ``Figure.savefig`` (also used by animation writers)."""

    def __init__(self, max_dpi: Optional[float]):
        self.max_dpi = max_dpi

    def __enter__(self):
        from matplotlib.figure import Figure
        import matplotlib

        self._orig = Figure.savefig
        cap = self.max_dpi
        if cap is None:
            return self
        orig = self._orig

        def savefig(fig, *args, **kwargs):
            dpi = kwargs.get("dpi")
            if dpi is None or dpi == "figure":
                dpi = matplotlib.rcParams["savefig.dpi"]
                if dpi == "figure":
                    dpi = fig.dpi
            kwargs["dpi"] = min(float(dpi), float(cap))
            return orig(fig, *args, **kwargs)

        Figure.savefig = savefig
        return self

    def __exit__(self, *exc):
        from matplotlib.figure import Figure

        Figure.savefig = self._orig
        return False


# (max width in px, keep every n-th frame) per quality; total playback time is preserved
GIF_SHRINK = {"preview": (560, 2), "standard": (900, 1), "original": (None, 1)}


def shrink_gif(data: bytes, max_width: Optional[int], frame_step: int = 1) -> bytes:
    """Downscale an animated GIF; with ``frame_step`` > 1 drop frames and stretch durations."""
    if not max_width and frame_step <= 1:
        return data
    from PIL import Image, ImageSequence

    src = Image.open(io.BytesIO(data))
    width = min(max_width or src.width, src.width)
    if width == src.width and frame_step <= 1:
        return data
    height = round(src.height * width / src.width)
    frames, durations = [], []
    for i, frame in enumerate(ImageSequence.Iterator(src)):
        duration = frame.info.get("duration", src.info.get("duration", 100))
        if i % frame_step:
            if durations:
                durations[-1] += duration
            continue
        durations.append(duration)
        frames.append(frame.convert("RGB").resize((width, height), Image.LANCZOS)
                      .quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=durations,
                   loop=src.info.get("loop", 0), optimize=True, disposal=2)
    return buf.getvalue()


def run_script(script: str, argv: Optional[List[str]] = None, out: Optional[Result] = None,
               captions: Optional[dict] = None, max_log_chars: int = 60000, quality: str = "preview",
               files: Optional[dict] = None) -> dict:
    """Execute ``script`` (path relative to the repo root) and add its outputs to ``out``.

    ``files`` maps relative paths to text written into the working directory first
    (e.g. the active dataset as CSV for ``main.py --mode analyze``).
    """
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
        inputs = set()
        for rel, text in (files or {}).items():
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            inputs.add(target)
        try:
            os.chdir(work)
            sys.argv = [script] + list(argv or [])
            if str(APP_ROOT) not in sys.path:
                sys.path.insert(0, str(APP_ROOT))
            if out is not None:
                out.progress(f"{script} läuft …")
            with _DpiCap(QUALITY.get(quality)), redirect_stdout(_Tee(buf, sys.__stdout__ if sys.__stdout__ else io.StringIO())):
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
        produced = sorted(p for p in work.rglob("*") if p.is_file() and p not in inputs)
        files = []
        for p in produced:
            rel = p.relative_to(work).as_posix()
            files.append({"path": rel, "bytes": p.stat().st_size})
            if out is None:
                continue
            suffix = p.suffix.lower()
            if suffix in IMAGE_TYPES:
                data = p.read_bytes()
                if suffix == ".gif":
                    data = shrink_gif(data, *GIF_SHRINK.get(quality, (None, 1)))
                out.image(data, captions.get(p.name, rel), mime=IMAGE_TYPES[suffix])
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


def script_tool(script: str, params: dict, out: Result, *, argv: Optional[List[str]] = None,
                files: Optional[dict] = None) -> dict:
    """Shared body of the ``script-*`` tools: run, then explain what happened."""
    res = run_script(script, argv, out, captions=CAPTIONS, quality=params.get("quality", "preview"), files=files)
    images = [f for f in res["files"] if Path(f["path"]).suffix.lower() in IMAGE_TYPES]
    if images:
        out.note(f"`{script}` lief unverändert im Browser und hat {len(images)} "
                 f"Abbildung{'en' if len(images) != 1 else ''} erzeugt – unten in voller Größe, per Klick vergrößerbar.",
                 "success")
    else:
        out.note(f"`{script}` lief durch, hat aber keine Abbildung erzeugt. Details in der Python-Ausgabe.", "warning")
    return res
