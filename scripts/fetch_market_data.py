#!/usr/bin/env python3
"""
Fetch daily market data from FRED and write ``site/data/market.json``.

The GitHub Pages workflow runs this on every deploy and on a daily schedule,
so the website always analyses current data.  If FRED is unreachable the
previous snapshot is kept (exit code 0) - a stale chart beats a dead site.

Series (FRED ids):  SP500 (last ~10 years, FRED licence window), NASDAQCOM,
VIXCLS (CBOE VIX), VXVCLS (CBOE 3-month VIX, for the term structure).

Usage:  python scripts/fetch_market_data.py [--out site/data/market.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SERIES = {
    "SP500": "S&P 500",
    "NASDAQCOM": "NASDAQ Composite",
    "VIXCLS": "CBOE VIX",
    "VXVCLS": "CBOE 3-month VIX",
}
# Datasets offered on the site: an index plus the volatility series aligned to it.
DATASETS = [
    {"id": "sp500", "name": "S&P 500", "index": "SP500", "vol": True,
     "blurb": "Leading US equity index, daily closing prices (FRED provides the last 10 years), with VIX and VIX3M."},
    {"id": "nasdaq", "name": "NASDAQ Composite", "index": "NASDAQCOM", "vol": True, "start": "1990-01-02",
     "blurb": "Technology-heavy US index since 1990, with VIX – covers the dot-com bubble, the financial crisis and COVID."},
    {"id": "nasdaq-long", "name": "NASDAQ Composite (since 1971)", "index": "NASDAQCOM", "vol": False,
     "blurb": "The longest available series: more than 50 years including the 1987 crash, without VIX."},
]
MIN_ROWS = 2000


def fetch_series(sid: str, retries: int = 4) -> dict:
    url = FRED_URL.format(sid=sid)
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tail-risk-lab/1.0 (+github pages build)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            rows = list(csv.reader(io.StringIO(text)))
            header, body = rows[0], rows[1:]
            if len(header) != 2:
                raise ValueError(f"unexpected header {header}")
            out = {}
            for date, value in body:
                value = value.strip()
                if value in ("", "."):
                    continue
                out[date[:10]] = float(value)
            if len(out) < MIN_ROWS:
                raise ValueError(f"only {len(out)} rows")
            return out
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"FRED {sid}: {last_error}")


def build_datasets(raw: dict) -> list:
    datasets = []
    for spec in DATASETS:
        index = raw[spec["index"]]
        dates = sorted(d for d in index if d >= spec.get("start", "0000"))
        if spec["vol"]:
            vix, vix3m = raw["VIXCLS"], raw.get("VXVCLS", {})
            dates = [d for d in dates if d in vix]
        entry = {
            "id": spec["id"], "name": spec["name"], "blurb": spec["blurb"],
            "source": f"FRED ({spec['index']}" + (", VIXCLS, VXVCLS)" if spec["vol"] else ")"),
            "dates": dates,
            "prices": [round(index[d], 4) for d in dates],
        }
        if spec["vol"]:
            entry["vix"] = [raw["VIXCLS"][d] for d in dates]
            # VIX3M starts in 2007 - pad with null so arrays stay aligned
            entry["vix3m"] = [vix3m.get(d) for d in dates]
        datasets.append(entry)
    return datasets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "site" / "data" / "market.json"))
    args = ap.parse_args()
    out = Path(args.out)
    try:
        raw = {sid: fetch_series(sid) for sid in SERIES}
        datasets = build_datasets(raw)
    except Exception as exc:  # noqa: BLE001
        if out.exists():
            print(f"WARNING: market data refresh failed ({exc}); keeping existing snapshot {out}", file=sys.stderr)
            return 0
        print(f"ERROR: market data fetch failed and no snapshot exists: {exc}", file=sys.stderr)
        return 1
    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Federal Reserve Bank of St. Louis, FRED (fred.stlouisfed.org)",
        "datasets": datasets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    for d in datasets:
        print(f"{d['id']:<12} {len(d['dates']):>6} rows  {d['dates'][0]} .. {d['dates'][-1]}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
