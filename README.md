# Analog session scanner

Finds the historically most-similar ES sessions to the one currently unfolding,
and projects their remainder onto today's chart. Also finds the most
*anti*-correlated sessions, since an inverted analog is the same computation
read from the other end of the sorted list.

Ships with a synthetic corpus so it runs before you have paid for any data.

## How it works

Each session is stored as 1380 one-minute bars indexed from the 18:00 ET
anchor, with the 17:00–18:00 maintenance break excluded. Bars are converted to
`log(close[i] / close[0])` before anything else touches them.

The whole corpus is a few MB of `Float32` and a full scan is a few hundred
thousand flops, so the matching runs in the browser. There is no database and
no serverless compute in the hot path — dragging the lookback slider re-scans
1250 sessions instantly.

## Getting data

Neither TradingView nor Schwab can supply the history. TradingView has no data
export API. Schwab has no futures bars at all and only ~30 days of 1-minute
equity data.

```bash
pip install databento
export DATABENTO_API_KEY=db-...
python scripts/backfill.py --start 2021-01-01 --end 2026-09-01
```

The script prints the exact cost from a free metadata call and asks before
spending anything. Load the resulting CSV with the "Load bars" button.

Any CSV with a timestamp column (`ts_event` / `timestamp` / `time`) and a
`close` column will load. ISO 8601 and bare epoch integers both work.

### Contract roll

`ES.c.0` is the continuous front month. Because matching happens on log returns
re-based to each session's own open, roll gaps never enter a within-day path.
You do not need a back-adjusted series.

## Live tail

The `/api/bars` route proxies Databento's *historical* endpoint, which serves
data older than roughly 24 hours. That covers replay and corpus building but not
the bar that just printed.

A true live tail needs a persistent process, which Vercel functions cannot host —
they cannot hold a socket open. The shape that works:

```
Databento Live  ->  small worker (Railway / Fly / VPS)  ->  Vercel Blob or KV
                                                              |
                                              /api/bars polls the store
```

Set `DATABENTO_API_KEY` in Vercel → Settings → Environment Variables.

## Reading the output honestly

Scanning ~1250 candidate days for the maximum correlation over a 120-bar window
will find something above 0.85 essentially every single day. That is a
multiple-comparisons artifact, not a signal.

Running the built-in backtest against the synthetic random-walk corpus makes
this concrete: mean best correlation lands near **0.89** while the mean forward
path correlation is **0.0002**. Excellent-looking matches, zero predictive
content.

Three guards are built in:

- **The fan, not the line.** Percentile bands across the top K show whether the
  analogs actually agree. A wide band is the honest answer.
- **The ambiguity flag.** When the best positive and best inverse analog are
  both near ±1, the window is too short to discriminate and the readout says so.
- **Walk-forward backtest.** Each test day only sees days strictly before it.
  Watch `forward path correlation` rather than the direction hit rate — hit rate
  over 250 days has a standard error above 3 points, so it will wander several
  points from the baselines on noise alone.

## Local development

```bash
npm install
npm run dev
```

## TradingView indicator

`tradingview/analog_session_projection.pine` is a Pine v6 port of the same idea,
scoped to what TradingView can actually do.

It cannot replace the web app. TradingView caps intraday chart history by plan
(20,000 bars on Premium), and a 23-hour ES session is 1380 one-minute bars, so
even Premium holds roughly 14 sessions on a 1-minute chart:

| Timeframe | Sessions at 20,000 bars |
|-----------|-------------------------|
| 1 min     | ~14                     |
| 5 min     | ~72                     |
| 15 min    | ~217                    |
| 30 min    | ~435                    |

What it is good for is the live half: it runs on your existing feed at no data
cost and updates on every tick.

Paths accumulate forward into a matrix as the script walks history, rather than
being read back with the `[]` operator, so `max_bars_back` is not the binding
constraint — the chart's loaded history is. Session slots come from elapsed
clock time rather than bar count, so a session with gaps still aligns against
one without them. Pine draws at most 500 bars into the future, which clips the
projection on long 1-minute sessions.

Set "Session length" to 23 for CME futures, 6.5 for RTH-only equities.

## Does the match survive forward?

`npm run persistence` matches at 12:00 ET and measures how well the projection
tracks reality at +30m, +1h, +2h and +4h. It runs a negative control (random
walks, no structure) and three positive controls (recurring archetype shapes at
increasing signal strength), so a null on real data can be distinguished from a
broken detector.

Add `-- --csv es_1m.csv` to run it on real bars.

Calibration from the synthetic controls, forward path correlation:

| horizon | noise | snr 0.3 | snr 0.6 | snr 0.9 |
|---------|-------|---------|---------|---------|
| +30m    | 0.018 |  0.000  |  0.050  |  0.401  |
| +1h     | 0.004 | -0.013  |  0.119  |  0.472  |
| +2h     | 0.027 | -0.043  |  0.133  |  0.456  |
| +4h     | 0.005 |  0.019  |  0.065  |  0.333  |

Three things fall out of this:

1. Mean in-sample match correlation is 0.849 on pure noise and 0.940 on strongly
   structured data. It barely moves. **Match quality tells you almost nothing
   about whether the projection will hold.**
2. At snr 0.3 — where 30% of every session is a genuinely recurring shape — the
   method extracts nothing. The detection threshold is high.
3. Persistence peaks around +1h to +2h and decays by +4h, even in the best case.
   The 12:00-to-close horizon is the hardest one to call.

Compare real results against the noise row, not against zero.
