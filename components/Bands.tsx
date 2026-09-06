"use client";

import { useCallback, useMemo, useState } from "react";
import { NormDay, Projection } from "../lib/engine";
import {
  BandEval,
  BandsModel,
  COVERAGES,
  bandProjection,
  evalBands,
  trainBands,
} from "../lib/bands";

export interface BandsState {
  model: BandsModel | null;
  training: boolean;
  coverage: number;
  setCoverage: (c: number) => void;
  touch: boolean;
  setTouch: (b: boolean) => void;
  train: (history: NormDay[], t: number) => void;
  ev: BandEval | null;
  scoring: boolean;
  score: (corpus: NormDay[], fromIdx: number, t: number) => void;
  projectionFor: (
    corpus: NormDay[],
    i: number,
    t: number
  ) => Projection | null;
  reset: () => void;
}

export function useBands(): BandsState {
  const [model, setModel] = useState<BandsModel | null>(null);
  const [training, setTraining] = useState(false);
  const [coverage, setCoverage] = useState(0.9);
  const [touch, setTouch] = useState(false);
  const [ev, setEv] = useState<BandEval | null>(null);
  const [scoring, setScoring] = useState(false);

  const train = useCallback((history: NormDay[], t: number) => {
    setTraining(true);
    setEv(null);
    setTimeout(() => {
      setModel(trainBands(history, t));
      setTraining(false);
    }, 20);
  }, []);

  const score = useCallback(
    (corpus: NormDay[], fromIdx: number, t: number) => {
      if (!model) return;
      setScoring(true);
      setTimeout(() => {
        setEv(evalBands(model, corpus, fromIdx, t));
        setScoring(false);
      }, 20);
    },
    [model]
  );

  const projectionFor = useCallback(
    (corpus: NormDay[], i: number, t: number) =>
      model ? bandProjection(model, corpus, i, t, coverage, touch) : null,
    [model, coverage, touch]
  );

  const reset = useCallback(() => {
    setModel(null);
    setEv(null);
  }, []);

  return useMemo(
    () => ({
      model,
      training,
      coverage,
      setCoverage,
      touch,
      setTouch,
      train,
      ev,
      scoring,
      score,
      projectionFor,
      reset,
    }),
    [model, training, coverage, touch, train, ev, scoring, score, projectionFor, reset]
  );
}

interface Props {
  bands: BandsState;
  corpus: NormDay[];
  history: NormDay[];
  targetIdx: number;
  t: number;
  unit: string;
  targetDate: string;
}

export function BandsPanel({
  bands,
  corpus,
  history,
  targetIdx,
  t,
  unit,
  targetDate,
}: Props) {
  const r = bands.model?.report;
  const stale = !!r && r.boundary >= targetDate;
  const f = bands.model?.predict(corpus, targetIdx, t) ?? null;
  const set = f ? (bands.touch ? f.touch : f.terminal) : null;

  return (
    <div className="panel">
      <h2>Range bands</h2>
      <p className="note">
        Forecasts how far price travels, not which way. Ridge on log forward
        volatility, then empirical quantiles of held-out standardized outcomes.
      </p>

      <button
        className="primary"
        onClick={() => bands.train(history, t)}
        disabled={bands.training || history.length < 80}
      >
        {bands.training ? "Fitting…" : `Fit on ${history.length} ${unit}`}
      </button>

      {history.length < 80 && (
        <p className="flag">Needs 80+ prior {unit}.</p>
      )}

      {r && (
        <>
          <div className="seg">
            <button
              className={!bands.touch ? "on" : ""}
              onClick={() => bands.setTouch(false)}
            >
              Settle
            </button>
            <button
              className={bands.touch ? "on" : ""}
              onClick={() => bands.setTouch(true)}
            >
              Never touch
            </button>
          </div>

          <div className="btns">
            {COVERAGES.map((c) => (
              <button
                key={c}
                className={bands.coverage === c ? "on" : ""}
                onClick={() => bands.setCoverage(c)}
              >
                {(c * 100).toFixed(0)}%
              </button>
            ))}
          </div>

          <dl className="bt">
            <div>
              <dt>Fitted through</dt>
              <dd>{r.boundary}</dd>
            </div>
            <div className="hi">
              <dt>Held-out R² (log σ)</dt>
              <dd>{r.r2Calib.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Naive rule R²</dt>
              <dd>{r.r2Naive.toFixed(3)}</dd>
            </div>
          </dl>

          {stale && (
            <p className="flag">
              Fitted through {r.boundary}, on or after {targetDate}. Refit from
              here before reading these levels.
            </p>
          )}

          {set && f && (
            <>
              <h2>Levels at {(bands.coverage * 100).toFixed(0)}%</h2>
              <ul className="matches">
                {set
                  .filter((s) => s.p === bands.coverage)
                  .map((s) => (
                    <li key={s.p}>
                      <span className="d">
                        {bands.touch ? "never touched" : "settled inside"}
                      </span>
                      <span className="c neg">{s.lo.toFixed(2)}</span>
                      <span className="c pos">{s.hi.toFixed(2)}</span>
                    </li>
                  ))}
                <li>
                  <span className="d">width</span>
                  <span className="v" />
                  <span className="v">
                    {(
                      ((set.find((s) => s.p === bands.coverage)!.hi -
                        set.find((s) => s.p === bands.coverage)!.lo) /
                        f.anchor) *
                      100
                    ).toFixed(2)}
                    %
                  </span>
                </li>
              </ul>
            </>
          )}

          <button
            onClick={() => bands.score(corpus, targetIdx, t)}
            disabled={bands.scoring || corpus.length - targetIdx < 30}
          >
            {bands.scoring
              ? "Scoring…"
              : `Check coverage on ${corpus.length - targetIdx} held-out ${unit}`}
          </button>

          {bands.ev && (
            <>
              <ul className="matches cov">
                <li className="hd">
                  <span className="d">target</span>
                  <span className="c">settle</span>
                  <span className="c">no touch</span>
                </li>
                {bands.ev.rows.map((row) => (
                  <li key={row.p}>
                    <span className="d">{(row.p * 100).toFixed(0)}%</span>
                    <span className="c pos">
                      {(row.terminalHit * 100).toFixed(1)}%
                    </span>
                    <span className="c">
                      {(row.touchHit * 100).toFixed(1)}%
                    </span>
                  </li>
                ))}
              </ul>
              <p className="flag">
                Measured on {bands.ev.n} {unit} the fit never saw. Settle
                coverage running above target means the bands are wide, which
                is the safe direction. Touch coverage is the one that bites:
                price wanders outside a level far more often than it closes
                outside one, so a strike chosen from the settle column will be
                breached intraday much more than its number suggests.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
