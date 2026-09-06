"use client";

import { useCallback, useMemo, useState } from "react";
import { NormDay, Projection } from "../lib/engine";
import {
  ModelEval,
  Trained,
  evalModel,
  modelProjection,
  trainModel,
} from "../lib/model";

export type Source = "analogs" | "model" | "bands";
export type Which = "ridge" | "mlp";

export interface Forecast {
  trained: Trained | null;
  training: boolean;
  which: Which;
  setWhich: (w: Which) => void;
  train: (history: NormDay[]) => void;
  ev: ModelEval | null;
  scoring: boolean;
  score: (testDays: NormDay[], t: number) => void;
  projectionFor: (day: NormDay, t: number) => Projection | null;
  reset: () => void;
}

/**
 * Training runs on the main thread and takes a few seconds, so it is always
 * behind an explicit button rather than firing on every cursor move. The
 * setTimeout is what lets the "Training…" label paint before the loop starts.
 */
export function useForecast(windowBars: number, anchored: boolean): Forecast {
  const [trained, setTrained] = useState<Trained | null>(null);
  const [training, setTraining] = useState(false);
  const [which, setWhich] = useState<Which>("ridge");
  const [ev, setEv] = useState<ModelEval | null>(null);
  const [scoring, setScoring] = useState(false);

  const train = useCallback(
    (history: NormDay[]) => {
      setTraining(true);
      setEv(null);
      setTimeout(() => {
        setTrained(trainModel(history, { windowBars, anchored }));
        setTraining(false);
      }, 20);
    },
    [windowBars, anchored]
  );

  const score = useCallback(
    (testDays: NormDay[], t: number) => {
      if (!trained) return;
      setScoring(true);
      setTimeout(() => {
        setEv(evalModel(trained, trained[which], testDays, t));
        setScoring(false);
      }, 20);
    },
    [trained, which]
  );

  const projectionFor = useCallback(
    (day: NormDay, t: number) =>
      trained ? modelProjection(trained, trained[which], day, t) : null,
    [trained, which]
  );

  const reset = useCallback(() => {
    setTrained(null);
    setEv(null);
  }, []);

  return useMemo(
    () => ({
      trained,
      training,
      which,
      setWhich,
      train,
      ev,
      scoring,
      score,
      projectionFor,
      reset,
    }),
    [trained, training, which, train, ev, scoring, score, projectionFor, reset]
  );
}

interface PanelProps {
  fc: Forecast;
  history: NormDay[];
  testDays: NormDay[];
  t: number;
  /** "sessions" or "weeks" — only affects wording. */
  unit: string;
  /** Date of the session being viewed, to detect a stale training boundary. */
  targetDate: string;
}

export function ForecastPanel({
  fc,
  history,
  testDays,
  t,
  unit,
  targetDate,
}: PanelProps) {
  const r = fc.trained?.report;
  // Trained on data at or after what is on screen: the prediction has seen
  // its own answer. Retraining is the only fix.
  const stale = !!r && r.boundary >= targetDate;

  return (
    <div className="panel">
      <h2>Model</h2>
      <p className="note">
        Fits a function from the window shape to the remaining path, rather
        than retrieving similar past {unit}. Trained only on {unit} before the
        one on screen.
      </p>

      <button
        className="primary"
        onClick={() => fc.train(history)}
        disabled={fc.training || history.length < 60}
      >
        {fc.training ? "Training…" : `Train on ${history.length} ${unit}`}
      </button>

      {history.length < 60 && (
        <p className="flag">
          Needs 60+ prior {unit}. Step forward to a later date.
        </p>
      )}

      {r && (
        <>
          <div className="seg">
            <button
              className={fc.which === "ridge" ? "on" : ""}
              onClick={() => fc.setWhich("ridge")}
            >
              Ridge
            </button>
            <button
              className={fc.which === "mlp" ? "on" : ""}
              onClick={() => fc.setWhich("mlp")}
            >
              Neural net
            </button>
          </div>

          <dl className="bt">
            <div>
              <dt>Trained through</dt>
              <dd>{r.boundary}</dd>
            </div>
            <div>
              <dt>Rows (train / val)</dt>
              <dd>
                {r.trainRows.toLocaleString()} / {r.valRows.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt>Stopped at epoch</dt>
              <dd>
                {r.epochs} of {r.stoppedAt}
              </dd>
            </div>
            <div>
              <dt>Ridge val R²</dt>
              <dd>{r.ridgeValR2.toFixed(4)}</dd>
            </div>
            <div>
              <dt>Net val R²</dt>
              <dd>{r.mlpValR2.toFixed(4)}</dd>
            </div>
          </dl>

          {stale && (
            <p className="flag">
              Trained through {r.boundary}, which is on or after {targetDate}.
              This prediction has seen its own outcome. Retrain from here.
            </p>
          )}

          <button
            onClick={() => fc.score(testDays, t)}
            disabled={fc.scoring || testDays.length < 20}
          >
            {fc.scoring ? "Scoring…" : `Score on ${testDays.length} held-out ${unit}`}
          </button>

          {fc.ev && (
            <>
              <dl className="bt">
                <div>
                  <dt>{unit[0].toUpperCase() + unit.slice(1)} scored</dt>
                  <dd>{fc.ev.n}</dd>
                </div>
                <div>
                  <dt>Direction hit rate</dt>
                  <dd>{(fc.ev.hitRate * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Always-long baseline</dt>
                  <dd>{(fc.ev.baselineLong * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Momentum baseline</dt>
                  <dd>{(fc.ev.baselineMomentum * 100).toFixed(1)}%</dd>
                </div>
                <div className="hi">
                  <dt>Forward path correlation</dt>
                  <dd>{fc.ev.meanPathCorr.toFixed(4)}</dd>
                </div>
              </dl>
              <p className="flag">
                Same definitions as the analog backtest, so the two numbers are
                directly comparable. A validation R² at or below zero means the
                fit is not beating the mean of the training data, and the band
                on the chart is then the only part still telling you something
                — it is the spread of held-out errors, not a forecast.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
