#!/usr/bin/env python3
"""Build canonical ARPM trades CSV from C3 snapshot JSONL + SQLite resolutions.

Snapshots must be the *orderbook* JSONL files named YYYY-MM-DD.jsonl (not trades_*.jsonl).
Resolutions come from market_data.db (table `resolutions`) when available.

Usage:
  python scripts/build_arpm_dataset_c3_jsonl.py \\
    --snapshots-dir /root/polymarket-arb-bot/c3_snapshots \\
    --resolutions-db /root/research_data/binance_vm82/market_data.db \\
    --since 2026-03-12 \\
    --out /root/auto-research-prediction-markets/data/arpm_c3_resolved.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


def load_resolutions(db_path: Path) -> dict[str, float]:
    import sqlite3

    con = sqlite3.connect(str(db_path))
    cur = con.execute("SELECT window_slug, outcome FROM resolutions")
    out: dict[str, float] = {}
    for slug, oc in cur:
        if oc is None:
            continue
        o = str(oc).strip().upper()
        if o == "UP":
            out[str(slug)] = 1.0
        elif o == "DOWN":
            out[str(slug)] = 0.0
    con.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path, required=True)
    ap.add_argument("--resolutions-db", type=Path, required=True)
    ap.add_argument("--since", default="2026-03-12", help="YYYY-MM-DD inclusive")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    since = args.since
    res = load_resolutions(args.resolutions_db)
    day_re = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_files = 0

    with args.out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["timestamp", "market_id", "price_yes", "outcome"])
        w.writeheader()

        for path in sorted(args.snapshots_dir.glob("*.jsonl")):
            m = day_re.match(path.name)
            if not m:
                continue
            day = m.group(1)
            if day < since:
                continue
            n_files += 1
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    w_ts = d.get("w")
                    p = d.get("p")
                    ts_ms = d.get("ts")
                    if w_ts is None or p is None or ts_ms is None:
                        continue
                    slug = f"btc-updown-5m-{int(w_ts)}"
                    if slug not in res:
                        continue
                    ts = pd.to_datetime(int(ts_ms), unit="ms", utc=True)
                    w.writerow(
                        {
                            "timestamp": ts.isoformat(),
                            "market_id": slug,
                            "price_yes": float(p),
                            "outcome": int(res[slug]),
                        }
                    )
                    n_rows += 1

    print(f"wrote {n_rows} rows from {n_files} snapshot files -> {args.out}")


if __name__ == "__main__":
    main()
