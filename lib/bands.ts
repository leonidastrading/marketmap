import { NormDay, Projection } from "./engine";

/**
 * Forecast how far price travels, not which way.
 *
 * Direction tested at zero in this codebase twice over -- analog retrieval and
 * a learned model both land on the baselines. Magnitude is a different
 * question: log window volatility correlates ~0.65 with log forward range on
 * the same data. Volatility clusters, direction does not.
 *
 * ------------------------------------------------------------------- method
 *
 * Two stages, because estimating extreme quantiles directly from ~1,300
 * sessions is hopeless -- a 2% quantile would rest on a couple of dozen
 * observations.
 *
 *   1. Scale.  Ridge regression on log(forward realized sigma) from volatility
 *      features. This is the part that carries the signal.
 *
 *   2. Shape.  Standardize held-out outcomes by their predicted scale, then
 *      take empirical quantiles of the pooled standardized residuals. Every
 *      held-out session contributes to every quantile, so the tail is
 *      estimated from the whole sample rather than from its own tail.
 *
 * The band for coverage p is then anchor * exp(z_q * predicted_scale), with
 * z_q read off the standardized distribution. Asymmetry is preserved: upper
 * and lower quantiles are taken separately rather than assuming a symmetric
 * move.
 *
 * ------------------------------------------------------------------- limits
 *
 * This forecasts REALIZED range. It says nothing about whether an option
 * priced against that range is cheap or expensive, because it has never seen
 * an option price. Implied volatility is not in this dataset, and the edge in
 * selling premium lives entirely in the gap between implied and realized.
 * A well-calibrated realized-range forecast is a necessary input to that
 * question and nowhere near a sufficient one.
 */

export interface BandOptions {
  /** Lookbacks in bars for the volatility features. */
  lookbacks: number[];
  l2: number;
}

export const DEFAULT_BAND_OPTIONS: BandOptions = {
  lookbacks: [30, 120, 390],
  l2: 1e-3,
};

/** Coverage levels offered to the UI. */
export const COVERAGES = [0.5, 0.8, 0.9, 0.95];

const NFEAT = 8;

function realizedSigma(r: Float32Array, from: number, to: number): number {
  let s = 0;
  let n = 0;
  for (let i = Math.max(1, from); i <= to; i++) {
    const d = r[i] - r[i - 1];
    s += d * d;
    n++;
  }
  return n > 0 ? Math.sqrt(s / n) : 0;
}

/**
 * Volatility features at (session i, bar t). Uses the previous session too --
 * vol clusters across the boundary, and yesterday is a strong prior for today.
 */
export function bandFeatures(
  corpus: NormDay[],
  i: number,
  t: number,
  opts: BandOptions
): Float32Array | null {
  const day = corpus[i];
  const n = day.r.length;
  if (t < 40 || t >= n - 10) return null;

  const x = new Float32Array(NFEAT);
  let k = 0;

  for (const lb of opts.lookbacks) {
    const s = realizedSigma(day.r, Math.max(1, t - lb + 1), t);
    if (!(s > 1e-12)) return null;
    x[k++] = Math.log(s);
  }

  // Previous session's whole-day sigma: cross-session clustering.
  const prev = i > 0 ? corpus[i - 1] : day;
  const ps = realizedSigma(prev.r, 1, prev.r.length - 1);
  x[k++] = Math.log(Math.max(ps, 1e-12));

  // Session-to-date range so far, a jump-sensitive complement to sigma.
  let hi = -Infinity;
  let lo = Infinity;
  for (let j = 0; j <= t; j++) {
    if (day.r[j] > hi) hi = day.r[j];
    if (day.r[j] < lo) lo = day.r[j];
  }
  x[k++] = Math.log(Math.max(hi - lo, 1e-9));

  x[k++] = t / (n - 1);
  x[k++] = Math.log(n - t); // remaining bars

  const dow = new Date(`${day.date}T00:00:00Z`).getUTCDay();
  x[k++] = dow >= 1 && dow <= 5 ? (dow - 3) / 2 : 0;

  return x.every(Number.isFinite) ? x : null;
}

/** Forward realized sigma per bar over the remainder. This is the target. */
function forwardSigma(day: NormDay, t: number): number | null {
  const n = day.r.length;
  const s = realizedSigma(day.r, t + 1, n - 1);
  return s > 1e-12 ? s : null;
}

function ridgeFit(X: Float32Array[], y: number[], l2: number): Float32Array {
  const d = NFEAT + 1;
  const A = new Float64Array(d * d);
  const b = new Float64Array(d);
  const z = new Float64Array(d);
  for (let r = 0; r < X.length; r++) {
    for (let j = 0; j < NFEAT; j++) z[j] = X[r][j];
    z[NFEAT] = 1;
    for (let a = 0; a < d; a++) {
      if (z[a] === 0) continue;
      for (let c = 0; c < d; c++) A[a * d + c] += z[a] * z[c];
      b[a] += z[a] * y[r];
    }
  }
  for (let a = 0; a < NFEAT; a++) A[a * d + a] += l2 * X.length;

  // Gaussian elimination with partial pivoting.
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r = c + 1; r < d; r++)
      if (Math.abs(A[r * d + c]) > Math.abs(A[piv * d + c])) piv = r;
    if (piv !== c) {
      for (let j = 0; j < d; j++) {
        const tmp = A[c * d + j];
        A[c * d + j] = A[piv * d + j];
        A[piv * d + j] = tmp;
      }
      const tb = b[c];
      b[c] = b[piv];
      b[piv] = tb;
    }
    const p = A[c * d + c];
    if (Math.abs(p) < 1e-12) continue;
    for (let r = c + 1; r < d; r++) {
      const f = A[r * d + c] / p;
      if (!f) continue;
      for (let j = c; j < d; j++) A[r * d + j] -= f * A[c * d + j];
      b[r] -= f * b[c];
    }
  }
  const w = new Float32Array(d);
  for (let r = d - 1; r >= 0; r--) {
    let s = b[r];
    for (let j = r + 1; j < d; j++) s -= A[r * d + j] * w[j];
    const p = A[r * d + r];
    w[r] = Math.abs(p) < 1e-12 ? 0 : s / p;
  }
  return w;
}

function dot(w: Float32Array, x: Float32Array) {
  let s = w[NFEAT];
  for (let j = 0; j < NFEAT; j++) s += w[j] * x[j];
  return s;
}

function quantile(sorted: number[], q: number) {
  if (!sorted.length) return 0;
  const p = (sorted.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(p);
  const hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}

export interface BandForecast {
  anchor: number;
  /** Predicted sigma of the terminal log move over the remainder. */
  scale: number;
  /** Settlement bands: price is inside at the stated historical frequency. */
  terminal: { p: number; lo: number; hi: number }[];
  /** Touch bands: price never traded outside these at the stated frequency. */
  touch: { p: number; lo: number; hi: number }[];
}

export interface BandsModel {
  predict(corpus: NormDay[], i: number, t: number): BandForecast | null;
  report: BandReport;
}

export interface BandReport {
  trainN: number;
  calibN: number;
  boundary: string;
  /** R² of log forward sigma, in sample and held out. */
  r2Train: number;
  r2Calib: number;
  /** Held-out R² of the naive "window vol carries forward" rule. */
  r2Naive: number;
}

/**
 * Fit on sessions strictly before the caller's target.
 *
 * The last 30% become the calibration slice: the ridge never sees it, and the
 * standardized quantiles come only from it. Calibrating on training residuals
 * would produce bands that are too tight, which is the failure mode that
 * matters here -- an overconfident band is worse than no band.
 */
export function trainBands(
  corpus: NormDay[],
  t: number,
  optsIn: Partial<BandOptions> = {}
): BandsModel | null {
  const opts = { ...DEFAULT_BAND_OPTIONS, ...optsIn };
  if (corpus.length < 80) return null;

  const cut = Math.floor(corpus.length * 0.7);
  const Xtr: Float32Array[] = [];
  const ytr: number[] = [];

  for (let i = 0; i < cut; i++) {
    const x = bandFeatures(corpus, i, t, opts);
    const fs = x ? forwardSigma(corpus[i], t) : null;
    if (!x || fs === null) continue;
    Xtr.push(x);
    ytr.push(Math.log(fs));
  }
  if (Xtr.length < 50) return null;

  const w = ridgeFit(Xtr, ytr, opts.l2);

  const r2 = (X: Float32Array[], y: number[]) => {
    const m = y.reduce((a, b) => a + b, 0) / y.length;
    let ss = 0;
    let tot = 0;
    for (let i = 0; i < y.length; i++) {
      const d = y[i] - dot(w, X[i]);
      ss += d * d;
      tot += (y[i] - m) * (y[i] - m);
    }
    return tot > 0 ? 1 - ss / tot : 0;
  };

  // Calibration slice: standardized outcomes.
  const Xca: Float32Array[] = [];
  const yca: number[] = [];
  const zTerm: number[] = [];
  const zUp: number[] = [];
  const zDn: number[] = [];
  let naiveSS = 0;
  let naiveTot = 0;
  const caY: number[] = [];

  for (let i = cut; i < corpus.length; i++) {
    const day = corpus[i];
    const n = day.r.length;
    const x = bandFeatures(corpus, i, t, opts);
    const fs = x ? forwardSigma(day, t) : null;
    if (!x || fs === null || day.lastReal < n - 2) continue;

    Xca.push(x);
    yca.push(Math.log(fs));
    caY.push(Math.log(fs));

    const remaining = n - 1 - t;
    const scale = Math.exp(dot(w, x)) * Math.sqrt(remaining);
    if (!(scale > 1e-12)) continue;

    const base = day.r[t];
    zTerm.push((day.r[n - 1] - base) / scale);

    let hi = -Infinity;
    let lo = Infinity;
    for (let j = t; j < n; j++) {
      const v = day.r[j] - base;
      if (v > hi) hi = v;
      if (v < lo) lo = v;
    }
    zUp.push(hi / scale);
    zDn.push(lo / scale);

    // Naive rule: forward sigma equals trailing 120-bar sigma.
    const naive = Math.log(
      Math.max(realizedSigma(day.r, Math.max(1, t - 119), t), 1e-12)
    );
    naiveSS += (Math.log(fs) - naive) ** 2;
  }
  if (zTerm.length < 40) return null;

  const mCa = caY.reduce((a, b) => a + b, 0) / caY.length;
  for (const v of caY) naiveTot += (v - mCa) ** 2;

  zTerm.sort((a, b) => a - b);
  zUp.sort((a, b) => a - b);
  zDn.sort((a, b) => a - b);

  const report: BandReport = {
    trainN: Xtr.length,
    calibN: zTerm.length,
    boundary: corpus[corpus.length - 1].date,
    r2Train: r2(Xtr, ytr),
    r2Calib: r2(Xca, yca),
    r2Naive: naiveTot > 0 ? 1 - naiveSS / naiveTot : 0,
  };

  return {
    report,
    predict(c, i, tt) {
      const day = c[i];
      const n = day.r.length;
      const x = bandFeatures(c, i, tt, opts);
      if (!x) return null;
      const remaining = n - 1 - tt;
      if (remaining < 5) return null;
      const scale = Math.exp(dot(w, x)) * Math.sqrt(remaining);
      const anchor = day.closes[tt];
      if (!(scale > 1e-12) || !Number.isFinite(anchor)) return null;

      const terminal = COVERAGES.map((p) => ({
        p,
        lo: anchor * Math.exp(quantile(zTerm, (1 - p) / 2) * scale),
        hi: anchor * Math.exp(quantile(zTerm, (1 + p) / 2) * scale),
      }));
      // Touch: the low must clear the worst downside excursion, so read the
      // lower tail of the running minimum and the upper tail of the maximum.
      const touch = COVERAGES.map((p) => ({
        p,
        lo: anchor * Math.exp(quantile(zDn, 1 - p) * scale),
        hi: anchor * Math.exp(quantile(zUp, p) * scale),
      }));
      return { anchor, scale, terminal, touch };
    },
  };
}

export interface CoverageRow {
  p: number;
  terminalHit: number;
  touchHit: number;
  meanWidthPct: number;
}

export interface BandEval {
  n: number;
  rows: CoverageRow[];
}

/** Measure realized coverage on sessions the model never saw. */
export function evalBands(
  model: BandsModel,
  corpus: NormDay[],
  fromIdx: number,
  t: number
): BandEval {
  const hitT = COVERAGES.map(() => 0);
  const hitX = COVERAGES.map(() => 0);
  const width = COVERAGES.map(() => 0);
  let n = 0;

  for (let i = fromIdx; i < corpus.length; i++) {
    const day = corpus[i];
    const nb = day.r.length;
    if (day.lastReal < nb - 2) continue;
    const f = model.predict(corpus, i, t);
    if (!f) continue;
    n++;

    const settle = day.closes[nb - 1];
    let hi = -Infinity;
    let lo = Infinity;
    for (let j = t; j < nb; j++) {
      const v = day.closes[j];
      if (v > hi) hi = v;
      if (v < lo) lo = v;
    }

    for (let k = 0; k < COVERAGES.length; k++) {
      const T = f.terminal[k];
      const X = f.touch[k];
      if (settle >= T.lo && settle <= T.hi) hitT[k]++;
      if (lo >= X.lo && hi <= X.hi) hitX[k]++;
      width[k] += ((T.hi - T.lo) / f.anchor) * 100;
    }
  }

  return {
    n,
    rows: COVERAGES.map((p, k) => ({
      p,
      terminalHit: n ? hitT[k] / n : 0,
      touchHit: n ? hitX[k] / n : 0,
      meanWidthPct: n ? width[k] / n : 0,
    })),
  };
}

/**
 * Render the forecast as the Projection the chart already draws.
 *
 * The band is widened over the remainder as sqrt(elapsed), which assumes the
 * standardized shape is self-similar across horizons. The quantiles were
 * calibrated at the terminal horizon, so the cone is exact at the right edge
 * and an approximation in between -- fine for placing strikes, not a claim
 * about intraday touch probability at every minute.
 *
 * Outer band is `p` coverage, inner band is the 50% level.
 */
export function bandProjection(
  model: BandsModel,
  corpus: NormDay[],
  i: number,
  t: number,
  coverage: number,
  touch: boolean
): Projection | null {
  const day = corpus[i];
  const n = day.r.length;
  const f = model.predict(corpus, i, t);
  if (!f) return null;

  const set = touch ? f.touch : f.terminal;
  const outer = set.find((s) => s.p === coverage) ?? set[set.length - 1];
  const inner = set.find((s) => s.p === 0.5) ?? outer;

  const remaining = n - 1 - t;
  if (remaining < 2) return null;

  // Convert each level back to a standardized z so it can be re-scaled along
  // the cone rather than held flat.
  const zOf = (price: number) => Math.log(price / f.anchor) / f.scale;
  const zLoOuter = zOf(outer.lo);
  const zHiOuter = zOf(outer.hi);
  const zLoInner = zOf(inner.lo);
  const zHiInner = zOf(inner.hi);

  const h = n - t;
  const mk = (z: number) => {
    const out = new Float32Array(h);
    for (let j = 0; j < h; j++) {
      const frac = Math.sqrt(j / remaining);
      out[j] = f.anchor * Math.exp(z * f.scale * frac);
    }
    return out;
  };

  const mid = new Float32Array(h).fill(f.anchor);
  const one = new Float32Array(1);
  one[0] = f.anchor;

  return {
    from: t,
    lines: [],
    mean: mid,
    p10: mk(zLoOuter),
    p25: mk(zLoInner),
    p50: mid,
    p75: mk(zHiInner),
    p90: mk(zHiOuter),
    backFrom: t,
    backLines: [],
    backP10: one,
    backP25: one,
    backP50: one,
    backP75: one,
    backP90: one,
  };
}
