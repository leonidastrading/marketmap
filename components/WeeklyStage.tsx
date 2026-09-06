"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Chart from "./Chart";
import {
  BacktestResult,
  DayPath,
  Metric,
  backtest,
  normalize,
  project,
  scan,
} from "../lib/engine";
import {
  DAY_BARS,
  WEEK_BARS,
  WeekMode,
  buildWeeks,
  fmtWeekSpan,
  weekBarLabel,
  weekMarks,
} from "../lib/week";
import { ForecastPanel, Source, useForecast } from "./Forecast";

/**
 * The weekly scanner: same engine, longer paths.
 *
 * Kept as its own component with its own state rather than a mode on the
 * daily stage, because the two are meant to be read side by side -- a week
 * that rhymes with 2023 while today rhymes with nothing is the interesting
 * case, and you cannot see that if one view replaces the other.
 */
export default function WeeklyStage({ days }: { days: DayPath[] }) {
  const [mode, setMode] = useState<WeekMode>("24h");
  const [idx, setIdx] = useState(0);
  const [fromBar, setFromBar] = useState(0);
  const [t, setT] = useState(0);
  const [locked, setLocked] = useState(true);
  const [topK, setTopK] = useState(15);
  const [metric, setMetric] = useState<Metric>("pearson");
  const [inverse, setInverse] = useState(false);
  const [showOutcome, setShowOutcome] = useState(false);
  const [showLines, setShowLines] = useState(true);
  const [bt, setBt] = useState<BacktestResult | null>(null);
  const [btRunning, setBtRunning] = useState(false);
  const [src, setSrc] = useState<Source>("analogs");

  const per = DAY_BARS[mode];
  const bars = WEEK_BARS[mode];
  // A model fitted on 6900-bar weeks is meaningless on 1950-bar weeks.
  const fcReset = useRef<(() => void) | null>(null);

  const built = useMemo(() => buildWeeks(days, mode), [days, mode]);
  const norm = useMemo(() => built.weeks.map(normalize), [built.weeks]);
  const marks = useMemo(() => weekMarks(mode), [mode]);

  // Bar indices mean different things in the two modes, so a mode change has
  // to reset the cursor rather than rescale it.
  useEffect(() => {
    setIdx(Math.max(0, built.weeks.length - 1));
    setFromBar(0);
    setT(DAY_BARS[mode] * 3); // Thursday open: match Mon-Wed, project Thu-Fri
    setLocked(true);
    setBt(null); // 24h and cash-only results are not comparable
    fcReset.current?.();
  }, [mode, built.weeks.length]);

  const safeIdx = Math.min(idx, Math.max(0, norm.length - 1));
  const target = norm[safeIdx];
  const history = useMemo(() => norm.slice(0, safeIdx), [norm, safeIdx]);

  const result = useMemo(() => {
    if (!target || history.length < 12) return null;
    return scan(target, history, {
      t,
      windowMinutes: t - fromBar + 1,
      anchored: locked && fromBar === 0,
      topK,
      metric,
      // Adjacent weeks share too much regime; 7 days drops the prior week.
      excludeNearDays: 7,
    });
  }, [target, history, t, fromBar, locked, topK, metric]);

  const matches = inverse ? result?.negative : result?.positive;

  const analogProjection = useMemo(() => {
    if (!target || !matches || matches.length === 0) return null;
    return project(target, history, matches, t, result?.lo ?? t);
  }, [target, history, matches, t, result]);

  const fc = useForecast(t - fromBar + 1, locked && fromBar === 0);
  fcReset.current = fc.reset;
  const projection =
    src === "model" && target ? fc.projectionFor(target, t) : analogProjection;

  if (norm.length === 0) {
    return (
      <section className="weekly">
        <h2 className="wk-title">Weekly</h2>
        <p className="flag">
          No complete Mon–Fri weeks in this corpus
          {built.incomplete > 0
            ? ` (${built.incomplete} partial weeks were skipped)`
            : ""}
          .
        </p>
      </section>
    );
  }

  function moveNow(next: number) {
    const n = Math.max(fromBar + 30, Math.min(bars - 30, next));
    if (!locked) setFromBar(Math.max(0, n - (t - fromBar)));
    setT(n);
  }
  function moveFrom(next: number) {
    setFromBar(Math.max(0, Math.min(t - 30, next)));
  }
  function runBacktest() {
    if (norm.length < 40) return;
    setBtRunning(true);
    setTimeout(() => {
      setBt(
        backtest(norm, {
          t,
          windowMinutes: t - fromBar + 1,
          anchored: locked && fromBar === 0,
          topK,
          metric,
          excludeNearDays: 7,
          startIndex: Math.floor(norm.length * 0.5),
          maxDays: 400,
        })
      );
      setBtRunning(false);
    }, 20);
  }

  function stepWeek(n: number) {
    setIdx(Math.max(0, Math.min(norm.length - 1, safeIdx + n)));
  }

  const spanBars = t - fromBar + 1;
  const thin = history.length < 12;

  return (
    <section className="weekly">
      <div className="wk-head">
        <h2 className="wk-title">Weekly</h2>
        <div className="seg">
          <button
            className={mode === "24h" ? "on" : ""}
            onClick={() => setMode("24h")}
          >
            24 hour
          </button>
          <button
            className={mode === "rth" ? "on" : ""}
            onClick={() => setMode("rth")}
          >
            Cash only
          </button>
        </div>
        <span className="note">
          {norm.length} weeks
          {built.incomplete > 0 && ` · ${built.incomplete} partial skipped`} ·{" "}
          {bars.toLocaleString()} bars/week
        </span>
      </div>

      <div className="stage">
        <div className="plot">
          <div className="chart chart-wk">
            <Chart
              target={target}
              t={t}
              projection={projection}
              inverse={inverse}
              showOutcome={showOutcome}
              showLines={showLines}
              windowLo={result?.lo ?? 0}
              marks={marks}
            />
          </div>

          <div className="scrubber">
            <div className="range2">
              <div className="track" />
              <div
                className="fill"
                style={{
                  left: `${(fromBar / (bars - 1)) * 100}%`,
                  right: `${100 - (t / (bars - 1)) * 100}%`,
                }}
              />
              <input
                aria-label="Week window start"
                type="range"
                min={0}
                max={bars - 1}
                value={fromBar}
                onChange={(e) => moveFrom(+e.target.value)}
              />
              <input
                aria-label="Week now"
                type="range"
                min={0}
                max={bars - 1}
                value={t}
                onChange={(e) => moveNow(+e.target.value)}
              />
            </div>
            <div className="scrubread">
              Window{" "}
              <b>
                {weekBarLabel(fromBar, mode)}&ndash;{weekBarLabel(t, mode)}{" "}
                &middot; {fmtWeekSpan(spanBars, mode)}
              </b>
            </div>
          </div>
        </div>

        <aside>
          <div className="row">
            <label>Forecast from</label>
            <div className="seg">
              <button
                className={src === "analogs" ? "on" : ""}
                onClick={() => setSrc("analogs")}
              >
                Analogs
              </button>
              <button
                className={src === "model" ? "on" : ""}
                onClick={() => setSrc("model")}
              >
                Model
              </button>
            </div>
          </div>

          <div className="row">
            <label>Week of</label>
            <div className="btns">
              <button onClick={() => stepWeek(-1)} disabled={safeIdx <= 0}>
                ‹ Prev
              </button>
              <input
                type="date"
                value={target.date}
                min={norm[0].date}
                max={norm[norm.length - 1].date}
                onChange={(e) => {
                  let best = -1;
                  for (let i = 0; i < norm.length; i++) {
                    if (norm[i].date <= e.target.value) best = i;
                    else break;
                  }
                  if (best >= 0) setIdx(best);
                }}
              />
              <button
                onClick={() => stepWeek(1)}
                disabled={safeIdx >= norm.length - 1}
              >
                Next ›
              </button>
            </div>
            <p className={thin ? "flag" : "note"}>
              {thin
                ? `Only ${history.length} weeks precede this one. Step forward for a usable corpus.`
                : `${history.length} weeks precede this one${
                    result ? `, ${result.candidates} eligible` : ""
                  }.`}
            </p>
          </div>

          <div className="row">
            <label>Cursor</label>
            <div className="btns">
              {[1, 2, 3, 4].map((d) => (
                <button
                  key={d}
                  className={t === d * per ? "on" : ""}
                  onClick={() => {
                    setLocked(true);
                    setFromBar(0);
                    setT(Math.min(bars - 30, d * per));
                  }}
                >
                  {["Mon", "Tue", "Wed", "Thu"][d - 1]}
                </button>
              ))}
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={locked}
                onChange={(e) => setLocked(e.target.checked)}
              />
              Match week to date
            </label>
            <p className="note">
              {locked && fromBar === 0
                ? "Matching everything from Monday's open to the cursor."
                : "Matching a trailing window that moves with the cursor."}
            </p>
          </div>

          <div className="row">
            <label htmlFor="wk">
              Analogs <b>{topK}</b>
            </label>
            <input
              id="wk"
              type="range"
              min={1}
              max={60}
              value={topK}
              onChange={(e) => setTopK(+e.target.value)}
            />
            <p className="note">
              The weekly corpus is ~5× smaller than the daily one, so a high
              count here reaches much further down the quality ranking.
            </p>
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
      </div>

      <div className="wk-readout">
        <ForecastPanel
          fc={fc}
          history={history}
          testDays={norm.slice(safeIdx)}
          t={t}
          unit="weeks"
          targetDate={target.date}
        />

        {result && matches && matches.length > 0 && (
          <div className="panel">
            <h2>
              Best {inverse ? "inverse " : ""}matches
            </h2>
            <ul className="matches">
              {matches.slice(0, 6).map((m) => (
                <li key={m.date}>
                  <span className="d">week of {m.date}</span>
                  <span className={`c ${m.inverse ? "neg" : "pos"}`}>
                    {m.corr.toFixed(3)}
                  </span>
                  <span className="v">×{m.volRatio.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="panel">
          <h2>Does it work?</h2>
          <p className="note">
            Walk-forward from the midpoint. Each test week sees only weeks that
            closed before it, at the same cursor and window as above.
          </p>
          <button
            className="primary"
            onClick={runBacktest}
            disabled={btRunning || norm.length < 40}
          >
            {btRunning ? "Running…" : "Run backtest"}
          </button>
          {norm.length < 40 && (
            <p className="flag">
              Needs 40+ weeks to split into train and test. This corpus has{" "}
              {norm.length}.
            </p>
          )}
          {bt && (
            <>
              <dl className="bt">
                <div>
                  <dt>Weeks tested</dt>
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
              <p className="flag">
                {bt.n} test weeks puts the standard error on hit rate near{" "}
                {((0.5 / Math.sqrt(bt.n)) * 100).toFixed(1)} points. A weekly
                corpus is inherently a fifth the size of the daily one over the
                same history, so this band is wide — treat anything inside it as
                indistinguishable from the baselines, and read the forward path
                correlation instead.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
