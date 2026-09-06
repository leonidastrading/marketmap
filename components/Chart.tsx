"use client";

import { useEffect, useRef } from "react";
import {
  NormDay,
  Projection,
  SESSION_BARS,
  RTH_OPEN_BAR,
  RTH_CLOSE_BAR,
} from "../lib/engine";

const C = {
  grid: "#1E2A38",
  gridStrong: "#2C3D4F",
  axis: "#6B7D91",
  today: "#F2B441",
  future: "#4A5866",
  pos: "#4CC9E8",
  neg: "#E8618C",
  band: "rgba(76, 201, 232, 0.10)",
  bandInner: "rgba(76, 201, 232, 0.18)",
  bandNeg: "rgba(232, 97, 140, 0.10)",
  bandNegInner: "rgba(232, 97, 140, 0.18)",
};

interface Props {
  target: NormDay;
  t: number;
  projection: Projection | null;
  inverse: boolean;
  showOutcome: boolean;
  showLines: boolean;
  windowLo: number;
}

export default function Chart({
  target,
  t,
  projection,
  inverse,
  showOutcome,
  showLines,
  windowLo,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const parent = cv.parentElement!;
    const dpr = window.devicePixelRatio || 1;
    const W = parent.clientWidth;
    const H = parent.clientHeight;
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.width = `${W}px`;
    cv.style.height = `${H}px`;
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const padL = 8;
    const padR = 62;
    const padT = 14;
    const padB = 26;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // ---- y range across everything that will be drawn -------------------
    let lo = Infinity;
    let hi = -Infinity;
    const bump = (v: number) => {
      if (Number.isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    };
    for (let i = 0; i <= Math.min(t, target.lastReal); i++) bump(target.closes[i]);
    if (showOutcome) {
      for (let i = t; i < SESSION_BARS; i++) bump(target.closes[i]);
    }
    if (projection) {
      for (let j = 0; j < projection.p10.length; j++) {
        bump(projection.p10[j]);
        bump(projection.p90[j]);
      }
      for (let j = 0; j < projection.backP10.length; j++) {
        bump(projection.backP10[j]);
        bump(projection.backP90[j]);
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
    const pad = (hi - lo) * 0.08 || 1;
    lo -= pad;
    hi += pad;

    const X = (bar: number) => padL + (bar / (SESSION_BARS - 1)) * plotW;
    const Y = (p: number) => padT + (1 - (p - lo) / (hi - lo)) * plotH;

    // ---- grid ------------------------------------------------------------
    g.font =
      '11px "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif';
    g.textBaseline = "middle";

    const ticks = 6;
    g.strokeStyle = C.grid;
    g.lineWidth = 1;
    for (let i = 0; i <= ticks; i++) {
      const p = lo + ((hi - lo) * i) / ticks;
      const y = Math.round(Y(p)) + 0.5;
      g.beginPath();
      g.moveTo(padL, y);
      g.lineTo(padL + plotW, y);
      g.stroke();
      g.fillStyle = C.axis;
      g.textAlign = "left";
      g.fillText(p.toFixed(2), padL + plotW + 8, y);
    }

    // Session landmarks carry more information than evenly spaced x ticks.
    const marks: [number, string][] = [
      [0, "18:00"],
      [360, "00:00"],
      [RTH_OPEN_BAR, "09:30"],
      [RTH_CLOSE_BAR, "16:00"],
    ];
    g.textAlign = "center";
    for (const [bar, label] of marks) {
      const x = Math.round(X(bar)) + 0.5;
      g.strokeStyle = bar === RTH_OPEN_BAR ? C.gridStrong : C.grid;
      g.beginPath();
      g.moveTo(x, padT);
      g.lineTo(x, padT + plotH);
      g.stroke();
      g.fillStyle = C.axis;
      // The first mark sits on the left edge, so centring it clips the label.
      g.textAlign = bar === 0 ? "left" : "center";
      g.fillText(label, x, padT + plotH + 13);
    }
    g.textAlign = "center";

    // ---- matching window shading ----------------------------------------
    g.fillStyle = "rgba(242, 180, 65, 0.06)";
    g.fillRect(X(windowLo), padT, X(t) - X(windowLo), plotH);

    const line = (
      pts: (bar: number) => [number, number],
      from: number,
      to: number,
      stroke: string,
      width: number,
      alpha = 1
    ) => {
      g.globalAlpha = alpha;
      g.strokeStyle = stroke;
      g.lineWidth = width;
      g.lineJoin = "round";
      g.beginPath();
      let started = false;
      const step = Math.max(1, Math.floor((to - from) / plotW));
      for (let i = from; i <= to; i += step) {
        const [x, y] = pts(i);
        if (!Number.isFinite(y)) continue;
        started ? g.lineTo(x, y) : g.moveTo(x, y);
        started = true;
      }
      g.stroke();
      g.globalAlpha = 1;
    };

    // ---- projection ------------------------------------------------------
    if (projection) {
      const p = projection;
      const h = p.p10.length;
      const bandFill = (
        a: Float32Array,
        b: Float32Array,
        fill: string
      ) => {
        g.fillStyle = fill;
        g.beginPath();
        for (let j = 0; j < h; j++) g.lineTo(X(t + j), Y(a[j]));
        for (let j = h - 1; j >= 0; j--) g.lineTo(X(t + j), Y(b[j]));
        g.closePath();
        g.fill();
      };
      bandFill(p.p10, p.p90, inverse ? C.bandNeg : C.band);
      bandFill(p.p25, p.p75, inverse ? C.bandNegInner : C.bandInner);

      // Backward half. Same anchor at t, so this converges to a point at the
      // "now" line and opens leftward — the width at any bar is the spread of
      // the analogs against a window they were selected to fit, which is the
      // honest picture of how good the match actually is.
      const bh = p.backP10.length;
      const backBand = (a: Float32Array, b: Float32Array, fill: string) => {
        g.fillStyle = fill;
        g.beginPath();
        for (let j = 0; j < bh; j++) g.lineTo(X(p.backFrom + j), Y(a[j]));
        for (let j = bh - 1; j >= 0; j--) g.lineTo(X(p.backFrom + j), Y(b[j]));
        g.closePath();
        g.fill();
      };
      backBand(p.backP10, p.backP90, inverse ? C.bandNeg : C.band);
      backBand(p.backP25, p.backP75, inverse ? C.bandNegInner : C.bandInner);

      if (showLines) {
        for (let k = 0; k < p.lines.length; k++) {
          const l = p.lines[k];
          const stroke = l.inverse ? C.neg : C.pos;
          line((i) => [X(t + i), Y(l.path[i])], 0, h - 1, stroke, 1, 0.16);
          const back = p.backLines[k];
          if (back) {
            line(
              (i) => [X(p.backFrom + i), Y(back[i])],
              0,
              bh - 1,
              stroke,
              1,
              0.12
            );
          }
        }
      }

      // Median across the fitted window, dashed to match the forward median so
      // the two read as one curve hinged on "now".
      g.setLineDash([5, 4]);
      line(
        (i) => [X(p.backFrom + i), Y(p.backP50[i])],
        0,
        bh - 1,
        inverse ? C.neg : C.pos,
        1.4,
        0.75
      );
      g.setLineDash([]);

      // Ensemble median, dashed — dashed because it is a central tendency of
      // disagreeing paths, not a forecast.
      g.setLineDash([5, 4]);
      line(
        (i) => [X(t + i), Y(p.p50[i])],
        0,
        h - 1,
        inverse ? C.neg : C.pos,
        1.8
      );
      g.setLineDash([]);
    }

    // ---- what actually happened (replay only) ----------------------------
    if (showOutcome) {
      line(
        (i) => [X(i), Y(target.closes[i])],
        t,
        Math.min(target.lastReal, SESSION_BARS - 1),
        C.future,
        1.5
      );
    }

    // ---- today's realized path ------------------------------------------
    line(
      (i) => [X(i), Y(target.closes[i])],
      0,
      Math.min(t, target.lastReal),
      C.today,
      1.9
    );

    // "now" marker
    const nx = Math.round(X(t)) + 0.5;
    g.strokeStyle = C.today;
    g.globalAlpha = 0.5;
    g.setLineDash([3, 3]);
    g.beginPath();
    g.moveTo(nx, padT);
    g.lineTo(nx, padT + plotH);
    g.stroke();
    g.setLineDash([]);
    g.globalAlpha = 1;

    const py = Y(target.closes[t]);
    g.fillStyle = C.today;
    g.beginPath();
    g.arc(nx, py, 3, 0, Math.PI * 2);
    g.fill();
  }, [target, t, projection, inverse, showOutcome, showLines, windowLo]);

  return <canvas ref={ref} />;
}
