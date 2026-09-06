#!/usr/bin/env python3
"""
Collapse a Databento ohlcv-1m export into a single front-month series.

A `stype_in=parent` request (symbols=["ES.FUT"]) returns every listed
expiration plus every calendar spread, so several contracts carry bars for the
same minute. Loading that straight into the app interleaves them and the
"session" becomes a sawtooth between contracts trading dollars apart.

This picks one contract per session -- the one with the most volume, which is
the front month by definition -- and emits just the columns the app reads.

Contract changes only ever happen at a session boundary, never inside one, so
no path contains a roll gap. The engine re-bases each session to its own open
before matching, so the level jump between sessions is irrelevant.

    python scripts/prepare_databento.py in.csv.zst -o es_1m.csv

Reads .zst directly; plain .csv also works.
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SESSION_BARS = 1380
ANCHOR = 18 * 60  # 18:00 ET, session start
BREAK = 17 * 60  # 17:00 ET, session end


def open_maybe_zst(path):
    """Return a text stream, transparently decompressing .zst."""
    if not path.endswith(".zst"):
        return open(path, "r", encoding="utf-8", newline="")
    try:
        import zstandard as zstd
    except ImportError:
        sys.exit("Need zstandard for .zst input:  pip install zstandard")
    fh = open(path, "rb")
    reader = zstd.ZstdDecompressor().stream_reader(fh)
    return io.TextIOWrapper(reader, encoding="utf-8", newline="")


def parse_ts(raw):
    """Databento pretty_ts gives ISO-8601 with Z; bare epochs also accepted."""
    if raw.isdigit():
        n = int(raw)
        for lim, div in ((10**17, 10**9), (10**14, 10**6), (10**11, 10**3)):
            if n > lim:
                return datetime.fromtimestamp(n / div, timezone.utc)
        return datetime.fromtimestamp(n, timezone.utc)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def session_of(dt_utc):
    """Mirror of sessionSlot() in lib/data.ts. Returns (date, bar) or None."""
    et = dt_utc.astimezone(ET)
    minutes = et.hour * 60 + et.minute
    if BREAK <= minutes < ANCHOR:
        return None  # daily maintenance window
    if minutes >= ANCHOR:
        bar = minutes - ANCHOR
        day = et.toordinal() + 1  # evening belongs to next settlement date
    else:
        bar = minutes + (24 * 60 - ANCHOR)
        day = et.toordinal()
    if not (0 <= bar < SESSION_BARS):
        return None
    return datetime.fromordinal(day).strftime("%Y-%m-%d"), bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="es_1m.csv")
    args = ap.parse_args()

    # Pass 1 -- total volume per (session, symbol) to find each session's front.
    vol = defaultdict(int)
    kept_syms, spread_rows, total = set(), 0, 0
    with open_maybe_zst(args.src) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            total += 1
            sym = row["symbol"]
            if "-" in sym:  # calendar spread, not an outright
                spread_rows += 1
                continue
            slot = session_of(parse_ts(row["ts_event"]))
            if slot is None:
                continue
            kept_syms.add(sym)
            vol[(slot[0], sym)] += int(row["volume"] or 0)

    front = {}
    for (date, sym), v in vol.items():
        if v > front.get(date, (0, None))[0]:
            front[date] = (v, sym)
    front = {d: s for d, (_, s) in front.items()}

    # Pass 2 -- emit only the front contract's bars, one row per minute.
    written, skipped = 0, 0
    with open_maybe_zst(args.src) as fh, open(args.out, "w", newline="") as out:
        rd = csv.DictReader(fh)
        w = csv.writer(out)
        w.writerow(["ts_event", "close"])
        for row in rd:
            sym = row["symbol"]
            if "-" in sym:
                continue
            dt = parse_ts(row["ts_event"])
            slot = session_of(dt)
            if slot is None:
                skipped += 1
                continue
            if front.get(slot[0]) != sym:
                continue
            w.writerow([int(dt.timestamp()), row["close"].rstrip("0").rstrip(".")])
            written += 1

    rolls = 0
    days = sorted(front)
    for a, b in zip(days, days[1:]):
        if front[a] != front[b]:
            rolls += 1

    print(f"read      {total:>12,} rows ({spread_rows:,} spreads dropped)")
    print(f"outrights {len(kept_syms):>12,} contracts")
    print(f"sessions  {len(front):>12,}  {days[0]} -> {days[-1]}")
    print(f"rolls     {rolls:>12,}")
    print(f"wrote     {written:>12,} rows -> {args.out}"
          f"  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
