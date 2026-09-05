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
import { syntheticCorpus, parseBars } from "../lib/data";

function barLabel(bar: number) {
  const m = (bar + 18 * 60) % (24 * 60);
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export default function Page() {
  const [corpus, setCorpus] = useState<NormDay[] | null>(null);
  const [source, setSource] = useState("demo");
  const [error, setError] = useState<string | null>(null);

  const [targetIdx, setTargetIdx] = useState(0);
  const [t, setT] = useState(1020);
  const [windowBars, setWindowBars] = useState(120);
  const [anchored, setAnchored] = useState(false);
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
        const next = v + 5;
        return next >= SESSION_BARS - 30 ? SESSION_BARS - 30 : next;
      });
    }, 90);
    return () => clearInterval(id);
  }, [playing]);

  useEffect(() => {
    if (t >= SESSION_BARS - 30) setPlaying(false);
  }, [t]);

  const target = corpus?.[targetIdx] ?? null;
  const history = useMemo(
    () => (corpus ? corpus.slice(0, targetIdx) : []),
    [corpus, targetIdx]
  );

  const result = useMemo(() => {
    if (!target || history.length < 20) return null;
    return scan(target, history, {
      t,
      windowBars,
      anchored,
      topK,
      metric,
      excludeNearDays: 3,
    });
  }, [target, history, t, windowBars, anchored, topK, metric]);

  const matches = inverse ? result?.negative : result?.positive;

  const projection = useMemo(() => {
    if (!target || !matches || matches.length === 0) return null;
    return project(target, history, matches, t);
  }, [target, history, matches, t]);

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
    const spread = ((projection.p90[h] - projection.p10[h]) / anchor) * 100;
    const bestPos = result.positive[0]?.corr ?? 0;
    const bestNeg = result.negative[0]?.corr ?? 0;
    const ambiguous = Math.abs(bestPos) > 0.85 && Math.abs(bestNeg) > 0.85;
    return {
      agreement,
      spread,
      bestPos,
      bestNeg,
      ambiguous,
      direction: up > last - up ? "higher" : "lower",
    };
  }, [projection, target, result, t]);

  function loadCsv(file: File) {
    setError(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const days = parseBars(String(reader.result));
        if (days.length < 30) {
          setError(
            `Only ${days.length} sessions parsed. Need at least 30 to scan.`
          );
          return;
        }
        const norm = days.map(normalize);
        setCorpus(norm);
        setTargetIdx(norm.length - 1);
        setSource(`${file.name} · ${days.length} sessions`);
        setBt(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not parse that file.");
      }
    };
    reader.readAsText(file);
  }

  function runBacktest() {
    if (!corpus) return;
    setBtRunning(true);
    setTimeout(() => {
      const r = backtest(corpus, {
        t,
        windowBars,
        anchored,
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
          <button onClick={() => fileRef.current?.click()}>Load bars</button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => e.target.files?.[0] && loadCsv(e.target.files[0])}
          />
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="stage">
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

        <aside>
          <div className="row">
            <label htmlFor="day">Session</label>
            <select
              id="day"
              value={targetIdx}
              onChange={(e) => setTargetIdx(+e.target.value)}
            >
              {corpus.map((d, i) =>
                i > 40 ? (
                  <option key={d.date} value={i}>
                    {d.date}
                  </option>
                ) : null
              )}
            </select>
          </div>

          <div className="row">
            <label htmlFor="t">
              Now <b>{barLabel(t)}</b>
            </label>
            <input
              id="t"
              type="range"
              min={60}
              max={SESSION_BARS - 30}
              value={t}
              onChange={(e) => setT(+e.target.value)}
            />
            <div className="btns">
              <button onClick={() => setPlaying((p) => !p)}>
                {playing ? "Pause" : "Replay"}
              </button>
              <button onClick={() => setT(RTH_OPEN_BAR)}>Cash open</button>
            </div>
          </div>

          <div className="row">
            <label htmlFor="w">
              Lookback <b>{anchored ? "session to date" : `${windowBars} bars`}</b>
            </label>
            <input
              id="w"
              type="range"
              min={20}
              max={480}
              step={10}
              value={windowBars}
              disabled={anchored}
              onChange={(e) => setWindowBars(+e.target.value)}
            />
            <label className="check">
              <input
                type="checkbox"
                checked={anchored}
                onChange={(e) => setAnchored(e.target.checked)}
              />
              Match from session open instead
            </label>
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
                  <dd>{diag.spread.toFixed(2)}%</dd>
                </div>
              </dl>
              <p className={diag.ambiguous || diag.agreement < 0.6 ? "flag" : "note"}>
                {diag.ambiguous
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
