"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Chart from "../components/Chart";
import {
  normalize,
  scan,
  project,
  backtest,
  Metric,
  NormDay,
  BacktestResult,
  SESSION_BARS,
  RTH_OPEN_BAR,
} from "../lib/engine";
import { syntheticCorpus, parseBars, fetchBars } from "../lib/data";

/** 90 -> "1h 30m". Traders think in clock time, not bar counts. */
function fmtDur(mins: number) {
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const r = mins % 60;
  return r ? `${h}h ${r}m` : `${h}h`;
}

function barLabel(bar: number) {
  const m = (bar + 18 * 60) % (24 * 60);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export default function Page() {
  const [corpus, setCorpus] = useState<NormDay[] | null>(null);
  const [source, setSource] = useState("demo");
  const [error, setError] = useState<string | null>(null);
  const [fetchStart, setFetchStart] = useState("2025-09-01");
  const [fetchEnd, setFetchEnd] = useState("2026-09-01");
  const [fetching, setFetching] = useState(false);

  const [targetIdx, setTargetIdx] = useState(0);
  const [t, setT] = useState(1020);
  const [fromBar, setFromBar] = useState(900);
  const [locked, setLocked] = useState(false);
  const [topK, setTopK] = useState(25);
  const [metric, setMetric] = useState<Metric>("pearson");
  const [inverse, setInverse] = useState(false);
  const [showOutcome, setShowOutcome] = useState(false);
  const [showLines, setShowLines] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [bt, setBt] = useState<BacktestResult | null>(null);
  const [btRunning, setBtRunning] = useState(false);

  // Demo corpus so the tool is usable before real data lands.
  useEffect(() => {
    const raw = syntheticCorpus(1250);
    const norm = raw.map(normalize);
    setCorpus(norm);
    setTargetIdx(norm.length - 1);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setT((v) => {
        const next = Math.min(v + 5, SESSION_BARS - 30);
        if (!locked) setFromBar((f) => Math.max(0, f + (next - v)));
        return next;
      });
    }, 90);
    return () => clearInterval(id);
  }, [playing, locked]);

  useEffect(() => {
    if (t >= SESSION_BARS - 30) setPlaying(false);
  }, [t]);

  const windowMins = t - fromBar + 1;

  // Unlocked, the start trails "now" and the window keeps its length.
  // Locked, the start stays put and the window grows as the session runs.
  function moveNow(next: number) {
    const n = Math.max(fromBar + 15, Math.min(SESSION_BARS - 30, next));
    if (!locked) setFromBar(Math.max(0, n - (t - fromBar)));
    setT(n);
  }
  function moveFrom(next: number) {
    setFromBar(Math.max(0, Math.min(t - 15, next)));
  }
  function preset(mins: number) {
    setLocked(false);
    setFromBar(Math.max(0, t - mins + 1));
  }

  const target = corpus?.[targetIdx] ?? null;
  const history = useMemo(
    () => (corpus ? corpus.slice(0, targetIdx) : []),
    [corpus, targetIdx]
  );

  const result = useMemo(() => {
    if (!target || history.length < 20) return null;
    return scan(target, history, {
      t,
      windowMinutes: windowMins,
      anchored: false,
      topK,
      metric,
      excludeNearDays: 3,
    });
  }, [target, history, t, windowMins, topK, metric]);

  const matches = inverse ? result?.negative : result?.positive;

  const projection = useMemo(() => {
    if (!target || !matches || matches.length === 0) return null;
    return project(target, history, matches, t, result?.lo ?? t);
  }, [target, history, matches, t, result]);

  // --- honesty metrics ------------------------------------------------------
  const diag = useMemo(() => {
    if (!projection || !target || !result) return null;
    const last = projection.lines.length;
    const anchor = target.closes[t];
    let up = 0;
    for (const l of projection.lines) {
      if (l.path[l.path.length - 1] > anchor) up++;
    }
    const agreement = last ? Math.max(up, last - up) / last : 0;
    const h = projection.p10.length - 1;
    // With a handful of analogs the 10th and 90th percentile collapse toward
    // the same value, so the number reads as certainty rather than as too
    // small a sample.
    const spread =
      last >= 5 ? ((projection.p90[h] - projection.p10[h]) / anchor) * 100 : null;
    const bestPos = result.positive[0]?.corr ?? 0;
    const bestNeg = result.negative[0]?.corr ?? 0;
    const ambiguous = Math.abs(bestPos) > 0.85 && Math.abs(bestNeg) > 0.85;
    return {
      agreement,
      spread,
      bestPos,
      bestNeg,
      ambiguous,
      thin: last < 5,
      direction: up > last - up ? "higher" : "lower",
    };
  }, [projection, target, result, t]);

  function loadDays(days: ReturnType<typeof parseBars>, label: string) {
    if (days.length < 30) {
      setError(`Only ${days.length} sessions parsed. Need at least 30 to scan.`);
      return;
    }
    const norm = days.map(normalize);
    setCorpus(norm);
    setTargetIdx(norm.length - 1);
    setSource(`${label} · ${days.length} sessions`);
    setBt(null);
  }

  /**
   * Read a CSV, transparently gunzipping when the name ends in .gz.
   *
   * A five-year 1-minute corpus is ~34 MB as text and ~7 MB gzipped, which is
   * the difference between a file you can move around and one you can't.
   * DecompressionStream streams it, so the compressed bytes are never held in
   * memory alongside the decompressed text.
   */
  async function readCsvText(file: File): Promise<string> {
    if (!/\.gz$/i.test(file.name)) return file.text();
    if (typeof DecompressionStream === "undefined") {
      throw new Error(
        "This browser cannot decompress .gz. Gunzip the file and load the .csv."
      );
    }
    const stream = file
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).text();
  }

  async function loadCsv(file: File) {
    setError(null);
    setFetching(true);
    try {
      loadDays(parseBars(await readCsvText(file)), file.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not parse that file.");
    } finally {
      setFetching(false);
    }
  }

  async function fetchReal() {
    setError(null);
    setFetching(true);
    try {
      const days = await fetchBars(fetchStart, fetchEnd);
      loadDays(days, `ES.c.0 ${fetchStart}→${fetchEnd}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Fetch failed.");
    } finally {
      setFetching(false);
    }
  }

  /** Move n sessions through the corpus, clamped. */
  function stepDay(n: number) {
    if (!corpus) return;
    setTargetIdx((i) => Math.max(0, Math.min(corpus.length - 1, i + n)));
  }

  /** Jump to the session on or immediately before an arbitrary date. */
  function gotoDate(date: string) {
    if (!corpus || !date) return;
    let best = -1;
    for (let i = 0; i < corpus.length; i++) {
      if (corpus[i].date <= date) best = i;
      else break;
    }
    if (best >= 0) setTargetIdx(best);
  }

  function runBacktest() {
    if (!corpus) return;
    setBtRunning(true);
    setTimeout(() => {
      const r = backtest(corpus, {
        t,
        windowMinutes: windowMins,
        anchored: false,
        topK,
        metric,
        excludeNearDays: 3,
        startIndex: Math.floor(corpus.length * 0.6),
        maxDays: 250,
      });
      setBt(r);
      setBtRunning(false);
    }, 20);
  }

  const fileRef = useRef<HTMLInputElement>(null);

  if (!corpus || !target) {
    return <div className="boot">Building sessions…</div>;
  }

  return (
    <main>
      <header>
        <h1>Analog session scanner</h1>
        <div className="meta">
          <span className={source === "demo" ? "warn" : "ok"}>
            {source === "demo"
              ? "Synthetic data — random walks, no real structure"
              : source}
          </span>
          <input
            type="date"
            aria-label="Fetch from"
            value={fetchStart}
            onChange={(e) => setFetchStart(e.target.value)}
          />
          <input
            type="date"
            aria-label="Fetch to"
            value={fetchEnd}
            onChange={(e) => setFetchEnd(e.target.value)}
          />
          <button onClick={fetchReal} disabled={fetching}>
            {fetching ? "Fetching…" : "Fetch ES"}
          </button>
          <button onClick={() => fileRef.current?.click()}>Load CSV</button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.gz,text/csv,application/gzip"
            hidden
            onChange={(e) => e.target.files?.[0] && loadCsv(e.target.files[0])}
          />
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="stage">
        <div className="plot">
          <div className="chart">
            <Chart
              target={target}
              t={t}
              projection={projection}
              inverse={inverse}
              showOutcome={showOutcome}
              showLines={showLines}
              windowLo={result?.lo ?? 0}
            />
          </div>

          {/*
            The scrubber is inset by the canvas's own padL/padR so a thumb sits
            directly beneath the bar it selects. Both inputs span the full bar
            range rather than their usable range — a range input maps its thumb
            across min..max, so a narrower max would compress the scale and
            break the alignment. moveFrom/moveNow clamp instead.
          */}
          <div className="scrubber">
            <div className="range2">
              <div className="track" />
              <div
                className="fill"
                style={{
                  left: `${(fromBar / (SESSION_BARS - 1)) * 100}%`,
                  right: `${100 - (t / (SESSION_BARS - 1)) * 100}%`,
                }}
              />
              <input
                aria-label="Window start"
                type="range"
                min={0}
                max={SESSION_BARS - 1}
                value={fromBar}
                onChange={(e) => moveFrom(+e.target.value)}
              />
              <input
                aria-label="Now"
                type="range"
                min={0}
                max={SESSION_BARS - 1}
                value={t}
                onChange={(e) => moveNow(+e.target.value)}
              />
            </div>
            <div className="scrubread">
              Window{" "}
              <b>
                {barLabel(fromBar)}&ndash;{barLabel(t)} &middot;{" "}
                {fmtDur(windowMins)}
              </b>
            </div>
          </div>
        </div>

        <aside>
          <div className="row">
            <label htmlFor="day">Session</label>
            <div className="btns">
              <button onClick={() => stepDay(-1)} disabled={targetIdx <= 0}>
                ‹ Prev
              </button>
              <input
                id="day"
                type="date"
                value={target.date}
                min={corpus[0].date}
                max={corpus[corpus.length - 1].date}
                onChange={(e) => gotoDate(e.target.value)}
              />
              <button
                onClick={() => stepDay(1)}
                disabled={targetIdx >= corpus.length - 1}
              >
                Next ›
              </button>
            </div>
            <p className={history.length < 20 ? "flag" : "note"}>
              {history.length < 20
                ? `Only ${history.length} sessions precede ${target.date}. Step forward, or load a corpus that starts earlier.`
                : `${history.length} sessions precede this one${
                    result ? `, ${result.candidates} eligible after filters` : ""
                  }. Nothing dated ${target.date} or later can enter the match.`}
            </p>
          </div>

          <div className="row">
            <label>Window controls</label>

            <label className="check">
              <input
                type="checkbox"
                checked={locked}
                onChange={(e) => setLocked(e.target.checked)}
              />
              Lock start in place
            </label>
            <p className="note">
              {locked
                ? "The start is pinned. Moving now, or replaying, grows the window."
                : "The start trails now, keeping the window the same length."}
            </p>

            <div className="btns">
              {[30, 60, 120, 240].map((m) => (
                <button
                  key={m}
                  className={windowMins === m && !locked ? "on" : ""}
                  onClick={() => preset(m)}
                >
                  {fmtDur(m)}
                </button>
              ))}
              <button
                className={fromBar === 0 ? "on" : ""}
                onClick={() => {
                  setFromBar(0);
                  setLocked(true);
                }}
              >
                Open
              </button>
            </div>

            <div className="btns">
              <button onClick={() => setPlaying((p) => !p)}>
                {playing ? "Pause" : "Replay"}
              </button>
              <button onClick={() => moveNow(RTH_OPEN_BAR)}>Cash open</button>
            </div>
          </div>

          <div className="row">
            <label htmlFor="k">
              Analogs <b>{topK}</b>
            </label>
            <input
              id="k"
              type="range"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(+e.target.value)}
            />
            {topK < 10 && (
              <p className="note">
                Below ~10 analogs the fan is mostly sampling noise.
              </p>
            )}
          </div>

          <div className="row">
            <label>Similarity</label>
            <div className="seg">
              <button
                className={metric === "pearson" ? "on" : ""}
                onClick={() => setMetric("pearson")}
              >
                Correlation
              </button>
              <button
                className={metric === "euclid" ? "on" : ""}
                onClick={() => setMetric("euclid")}
              >
                Distance
              </button>
            </div>
            <p className="note">
              {metric === "pearson"
                ? "Matches shape only — a 0.1% day can match a 3% day."
                : "Matches shape and magnitude on vol-normalised paths."}
            </p>
          </div>

          <div className="row checks">
            <label className="check">
              <input
                type="checkbox"
                checked={inverse}
                onChange={(e) => setInverse(e.target.checked)}
              />
              Show inverse analogs
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={showLines}
                onChange={(e) => setShowLines(e.target.checked)}
              />
              Show individual paths
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={showOutcome}
                onChange={(e) => setShowOutcome(e.target.checked)}
              />
              Reveal what actually happened
            </label>
          </div>
        </aside>
      </section>

      <section className="readout">
        <div className="panel">
          <h2>Read</h2>
          {diag && (
            <>
              <div className="stat">
                <span>{(diag.agreement * 100).toFixed(0)}%</span> of {topK}{" "}
                analogs close {diag.direction} than {target.closes[t].toFixed(2)}
              </div>
              <dl>
                <div>
                  <dt>Best match</dt>
                  <dd>{diag.bestPos.toFixed(3)}</dd>
                </div>
                <div>
                  <dt>Best inverse</dt>
                  <dd>{diag.bestNeg.toFixed(3)}</dd>
                </div>
                <div>
                  <dt>10–90 spread</dt>
                  <dd>
                    {diag.spread === null
                      ? `— (needs 5+ analogs)`
                      : `${diag.spread.toFixed(2)}%`}
                  </dd>
                </div>
              </dl>
              <p className={diag.ambiguous || diag.agreement < 0.6 || diag.thin ? "flag" : "note"}>
                {diag.thin
                  ? `${projection?.lines.length} analog${projection?.lines.length === 1 ? "" : "s"} is not a sample. One path agreeing with itself is not agreement — raise the count to at least 10.`
                  : diag.ambiguous
                    ? "The best positive and best inverse analogs are both near ±1. The window is too short to discriminate — this is curve fitting, not a signal."
                    : diag.agreement < 0.6
                      ? "The analogs disagree on direction. There is no directional read here, only a range."
                      : "Analogs broadly agree. Check the backtest before acting on it."}
              </p>
            </>
          )}
        </div>

        <div className="panel">
          <h2>Matches</h2>
          <ul className="matches">
            {matches?.slice(0, 12).map((m) => (
              <li key={m.date}>
                <span className="d">{m.date}</span>
                <span className={m.inverse ? "c neg" : "c pos"}>
                  {m.corr.toFixed(3)}
                </span>
                <span className="v">×{m.volRatio.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <h2>Does it work?</h2>
          <p className="note">
            Walk-forward from the midpoint of the corpus. Each test day only sees
            days that came before it.
          </p>
          <button className="primary" onClick={runBacktest} disabled={btRunning}>
            {btRunning ? "Running…" : "Run backtest"}
          </button>
          {bt && (
            <dl className="bt">
              <div>
                <dt>Days tested</dt>
                <dd>{bt.n}</dd>
              </div>
              <div>
                <dt>Direction hit rate</dt>
                <dd>{(bt.hitRate * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Always-long baseline</dt>
                <dd>{(bt.baselineLong * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Momentum baseline</dt>
                <dd>{(bt.baselineMomentum * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Mean best |corr|</dt>
                <dd>{bt.meanBestCorr.toFixed(3)}</dd>
              </div>
              <div className="hi">
                <dt>Forward path correlation</dt>
                <dd>{bt.meanPathCorr.toFixed(4)}</dd>
              </div>
            </dl>
          )}
          {bt && (
            <p className="flag">
              Forward path correlation is the number that matters. Hit rate on{" "}
              {bt.n} days has a standard error near{" "}
              {((0.5 / Math.sqrt(bt.n)) * 100).toFixed(1)} points, so anything
              inside a few points of the baselines is noise.
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
