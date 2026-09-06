"use client";

import { useEffect, useMemo, useState } from "react";
import Chart from "./Chart";
import { DayPath, Metric, normalize, project, scan } from "../lib/engine";
import {
  DAY_BARS,
  WEEK_BARS,
  WeekMode,
  buildWeeks,
  fmtWeekSpan,
  weekBarLabel,
  weekMarks,
} from "../lib/week";

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

  const per = DAY_BARS[mode];
  const bars = WEEK_BARS[mode];

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

  const projection = useMemo(() => {
    if (!target || !matches || matches.length === 0) return null;
    return project(target, history, matches, t, result?.lo ?? t);
  }, [target, history, matches, t, result]);

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

      {result && matches && matches.length > 0 && (
        <div className="wk-matches">
          <span className="note">
            Best {inverse ? "inverse " : ""}matches ·{" "}
            {mode === "24h" ? "24 hour" : "cash only"}
          </span>
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
    </section>
  );
}
