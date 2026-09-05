#!/usr/bin/env python3
"""
Pull ES 1-minute bars from Databento and write a CSV the app can load.

    pip install databento
    export DATABENTO_API_KEY=db-...
    python scripts/backfill.py --start 2021-01-01 --end 2026-09-01

Prints the exact cost first and asks before spending anything. metadata.get_cost
is a free call, so there is no reason to run a query blind.

Symbology note: ES.c.0 is the continuous front month under Databento's default
roll rule. Because the app matches on log returns from the session anchor rather
than on price levels, contract roll gaps do not contaminate the within-day
paths — each session is re-based to its own open. You do not need a
back-adjusted series for this.
"""

import argparse
import os
import sys

try:
    import databento as db
except ImportError:
    sys.exit("pip install databento")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--symbol", default="ES.c.0")
    ap.add_argument("--out", default="es_1m.csv")
    ap.add_argument("--yes", action="store_true", help="skip the cost prompt")
    args = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("Set DATABENTO_API_KEY first.")

    client = db.Historical(key)
    query = dict(
        dataset="GLBX.MDP3",
        symbols=[args.symbol],
        stype_in="continuous",
        schema="ohlcv-1m",
        start=args.start,
        end=args.end,
    )

    cost = client.metadata.get_cost(**query)
    size = client.metadata.get_billable_size(**query)
    print(f"Range   {args.start} -> {args.end}  ({args.symbol})")
    print(f"Size    {size / 1e6:.1f} MB billable")
    print(f"Cost    ${cost:.2f}")

    if not args.yes:
        if input("Fetch? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled. Nothing was charged.")

    print("Fetching…")
    data = client.timeseries.get_range(**query)
    df = data.to_df()

    # The app wants ts_event as an ISO-8601 UTC string plus a close column.
    df = df.reset_index()
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    out = df[[ts_col, "open", "high", "low", "close", "volume"]].copy()
    out = out.rename(columns={ts_col: "ts_event"})
    out["ts_event"] = out["ts_event"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(args.out, index=False)

    mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {len(out):,} bars to {args.out} ({mb:.1f} MB)")
    print("Load it with the 'Load bars' button in the app.")


if __name__ == "__main__":
    main()
