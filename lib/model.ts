import { NormDay, Projection } from "./engine";

/**
 * A learned alternative to analog matching.
 *
 * Instead of retrieving similar past paths, this fits a function from
 * "shape so far" to "shape remaining" and evaluates it out of sample.
 *
 * ------------------------------------------------------------------ sizing
 *
 * The corpus has ~1.8M one-minute bars, and that number is misleading. The
 * task is one forward path per session, so the independent sample count is
 * the session count -- about 1,300 for five years, not 1.8M. Minutes inside a
 * session are one correlated observation, not a thousand.
 *
 * Sampling several cursors per session gives the optimizer more rows to chew
 * on and teaches it to condition on time-of-day, but it does not manufacture
 * information: twelve cursors from one Tuesday are still one Tuesday. Model
 * capacity is set against the session count, which is why the default net is
 * small and ridge regression runs beside it. If the net cannot beat a linear
 * model on held-out data, the extra capacity is fitting noise, and that
 * comparison is the point of having both.
 *
 * ------------------------------------------------------------------ leakage
 *
 * Everything is date-ordered. Training rows come only from sessions strictly
 * before the evaluation boundary; feature standardization uses training rows
 * only; residual quantiles for the uncertainty band come from a validation
 * slice the optimizer never saw. Getting any of this wrong produces
 * spectacular results that mean nothing.
 */

export const FEAT_POINTS = 32; // resampled window shape
export const HORIZONS = 8; // forward points predicted

/** Features: shape + scale + clock. Extra length is dow one-hot. */
export const FEAT_DIM = FEAT_POINTS + 3 + 5;

export interface ModelOptions {
  /** Trailing window length in bars. Ignored when anchored. */
  windowBars: number;
  anchored: boolean;
  hidden: number[];
  epochs: number;
  lr: number;
  l2: number;
  cursorsPerSession: number;
  seed: number;
  /** Validation checks (every 5 epochs) without improvement before stopping. */
  patience: number;
}

export const DEFAULT_OPTIONS: ModelOptions = {
  windowBars: 120,
  anchored: false,
  hidden: [24, 12],
  epochs: 200,
  lr: 0.01,
  l2: 3e-3,
  cursorsPerSession: 12,
  seed: 7,
  // Validation loss bottoms out within a handful of epochs on a corpus this
  // size -- the net starts memorising almost immediately. Patience turns a
  // 23-second grind into a few seconds and changes none of the results.
  patience: 8,
};

// ------------------------------------------------------------------ helpers

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Linear resample of src[lo..hi] onto `n` points. */
function resample(src: Float32Array, lo: number, hi: number, n: number, out: Float32Array, at: number) {
  const span = hi - lo;
  for (let i = 0; i < n; i++) {
    const p = lo + (span * i) / (n - 1);
    const j = Math.floor(p);
    const f = p - j;
    const a = src[j];
    const b = j + 1 <= hi ? src[j + 1] : a;
    out[at + i] = a + (b - a) * f;
  }
}

function std(v: Float32Array, from: number, to: number): number {
  let m = 0;
  const n = to - from + 1;
  for (let i = from; i <= to; i++) m += v[i];
  m /= n;
  let s = 0;
  for (let i = from; i <= to; i++) {
    const d = v[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / Math.max(1, n - 1));
}

export interface Sample {
  x: Float32Array; // FEAT_DIM
  y: Float32Array; // HORIZONS, scaled log-returns
  scale: number; // path sd used to scale both
  date: string;
}

/**
 * Build one training/prediction row.
 *
 * Shape is divided by the window's own standard deviation, so the model sees
 * a scale-free path and cannot win by memorising 2022's volatility. The scale
 * is handed back separately as a feature and as the multiplier that turns a
 * prediction back into log-returns.
 */
export function makeFeatures(
  day: NormDay,
  t: number,
  windowBars: number,
  anchored: boolean
): { x: Float32Array; scale: number } | null {
  const n = day.r.length;
  if (t < 30 || t >= n - 2) return null;
  const lo = anchored ? 0 : Math.max(0, t - windowBars + 1);
  if (t - lo < 20) return null;

  const base = day.r[t];
  const shape = new Float32Array(t - lo + 1);
  for (let i = lo; i <= t; i++) shape[i - lo] = day.r[i] - base;

  const sd = std(shape, 0, shape.length - 1);
  if (!(sd > 1e-9)) return null;

  let barVol = 0;
  for (let i = 1; i < shape.length; i++) {
    const d = shape[i] - shape[i - 1];
    barVol += d * d;
  }
  barVol = Math.sqrt(barVol / Math.max(1, shape.length - 1));
  if (!(barVol > 1e-12)) return null;

  const x = new Float32Array(FEAT_DIM);
  const norm = new Float32Array(shape.length);
  for (let i = 0; i < shape.length; i++) norm[i] = shape[i] / sd;
  resample(norm, 0, shape.length - 1, FEAT_POINTS, x, 0);

  x[FEAT_POINTS] = Math.log(sd);
  x[FEAT_POINTS + 1] = Math.log(barVol);
  x[FEAT_POINTS + 2] = t / (n - 1);

  const dow = new Date(`${day.date}T00:00:00Z`).getUTCDay() - 1;
  if (dow >= 0 && dow < 5) x[FEAT_POINTS + 3 + dow] = 1;

  if (!x.every(Number.isFinite)) return null;
  return { x, scale: sd };
}

/** Forward targets at evenly spaced horizons, scaled the same way. */
function makeTarget(day: NormDay, t: number, scale: number): Float32Array | null {
  const n = day.r.length;
  const base = day.r[t];
  const y = new Float32Array(HORIZONS);
  for (let k = 0; k < HORIZONS; k++) {
    const idx = Math.round(t + ((n - 1 - t) * (k + 1)) / HORIZONS);
    if (idx <= t || idx > n - 1) return null;
    y[k] = (day.r[idx] - base) / scale;
  }
  return y.every(Number.isFinite) ? y : null;
}

export function buildSamples(
  corpus: NormDay[],
  opts: ModelOptions
): Sample[] {
  const rng = mulberry32(opts.seed);
  const out: Sample[] = [];
  for (const day of corpus) {
    const n = day.r.length;
    const lowT = Math.max(60, Math.floor(n * 0.15));
    const highT = Math.floor(n * 0.9);
    for (let c = 0; c < opts.cursorsPerSession; c++) {
      const t =
        opts.cursorsPerSession === 1
          ? Math.floor((lowT + highT) / 2)
          : Math.floor(lowT + rng() * (highT - lowT));
      const f = makeFeatures(day, t, opts.windowBars, opts.anchored);
      if (!f) continue;
      const y = makeTarget(day, t, f.scale);
      if (!y) continue;
      out.push({ x: f.x, y, scale: f.scale, date: day.date });
    }
  }
  return out;
}

// ------------------------------------------------------------- standardizer

interface Scaler {
  mean: Float32Array;
  sd: Float32Array;
}

function fitScaler(rows: Sample[]): Scaler {
  const mean = new Float32Array(FEAT_DIM);
  const sd = new Float32Array(FEAT_DIM);
  for (const r of rows) for (let j = 0; j < FEAT_DIM; j++) mean[j] += r.x[j];
  for (let j = 0; j < FEAT_DIM; j++) mean[j] /= rows.length;
  for (const r of rows)
    for (let j = 0; j < FEAT_DIM; j++) {
      const d = r.x[j] - mean[j];
      sd[j] += d * d;
    }
  for (let j = 0; j < FEAT_DIM; j++) {
    sd[j] = Math.sqrt(sd[j] / Math.max(1, rows.length - 1));
    if (!(sd[j] > 1e-8)) sd[j] = 1; // constant column
  }
  return { mean, sd };
}

function applyScaler(s: Scaler, x: Float32Array): Float32Array {
  const o = new Float32Array(FEAT_DIM);
  for (let j = 0; j < FEAT_DIM; j++) o[j] = (x[j] - s.mean[j]) / s.sd[j];
  return o;
}

// -------------------------------------------------------------------- ridge

/** Solve (X'X + lI)b = X'y by Gaussian elimination with partial pivoting. */
function solve(A: Float64Array, b: Float64Array, d: number): Float64Array {
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r = c + 1; r < d; r++)
      if (Math.abs(A[r * d + c]) > Math.abs(A[piv * d + c])) piv = r;
    if (piv !== c) {
      for (let k = 0; k < d; k++) {
        const tmp = A[c * d + k];
        A[c * d + k] = A[piv * d + k];
        A[piv * d + k] = tmp;
      }
      const tb = b[c];
      b[c] = b[piv];
      b[piv] = tb;
    }
    const p = A[c * d + c];
    if (Math.abs(p) < 1e-12) continue;
    for (let r = c + 1; r < d; r++) {
      const f = A[r * d + c] / p;
      if (f === 0) continue;
      for (let k = c; k < d; k++) A[r * d + k] -= f * A[c * d + k];
      b[r] -= f * b[c];
    }
  }
  const out = new Float64Array(d);
  for (let r = d - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < d; k++) s -= A[r * d + k] * out[k];
    const p = A[r * d + r];
    out[r] = Math.abs(p) < 1e-12 ? 0 : s / p;
  }
  return out;
}

/** Independent ridge per horizon, with an intercept column. */
function fitRidge(rows: Sample[], sc: Scaler, l2: number): Float32Array[] {
  const d = FEAT_DIM + 1;
  const XtX = new Float64Array(d * d);
  const Xty: Float64Array[] = Array.from(
    { length: HORIZONS },
    () => new Float64Array(d)
  );
  const z = new Float64Array(d);
  for (const r of rows) {
    const xs = applyScaler(sc, r.x);
    for (let j = 0; j < FEAT_DIM; j++) z[j] = xs[j];
    z[FEAT_DIM] = 1;
    for (let a = 0; a < d; a++) {
      const za = z[a];
      if (za === 0) continue;
      for (let b2 = 0; b2 < d; b2++) XtX[a * d + b2] += za * z[b2];
      for (let k = 0; k < HORIZONS; k++) Xty[k][a] += za * r.y[k];
    }
  }
  for (let a = 0; a < FEAT_DIM; a++) XtX[a * d + a] += l2 * rows.length;
  return Xty.map((y) => {
    const A = Float64Array.from(XtX);
    const sol = solve(A, Float64Array.from(y), d);
    return Float32Array.from(sol);
  });
}

function ridgePredict(w: Float32Array[], xs: Float32Array): Float32Array {
  const out = new Float32Array(HORIZONS);
  for (let k = 0; k < HORIZONS; k++) {
    let s = w[k][FEAT_DIM];
    for (let j = 0; j < FEAT_DIM; j++) s += w[k][j] * xs[j];
    out[k] = s;
  }
  return out;
}

// ---------------------------------------------------------------------- mlp

interface Layer {
  w: Float32Array;
  b: Float32Array;
  nin: number;
  nout: number;
  mw: Float32Array;
  vw: Float32Array;
  mb: Float32Array;
  vb: Float32Array;
}

function initLayers(dims: number[], rng: () => number): Layer[] {
  const ls: Layer[] = [];
  for (let i = 0; i < dims.length - 1; i++) {
    const nin = dims[i];
    const nout = dims[i + 1];
    const w = new Float32Array(nin * nout);
    const lim = Math.sqrt(6 / (nin + nout)); // Glorot
    for (let j = 0; j < w.length; j++) w[j] = (rng() * 2 - 1) * lim;
    ls.push({
      w,
      b: new Float32Array(nout),
      nin,
      nout,
      mw: new Float32Array(nin * nout),
      vw: new Float32Array(nin * nout),
      mb: new Float32Array(nout),
      vb: new Float32Array(nout),
    });
  }
  return ls;
}

function forward(ls: Layer[], x: Float32Array): Float32Array[] {
  const acts: Float32Array[] = [x];
  let cur = x;
  for (let li = 0; li < ls.length; li++) {
    const L = ls[li];
    const out = new Float32Array(L.nout);
    for (let o = 0; o < L.nout; o++) {
      let s = L.b[o];
      for (let i = 0; i < L.nin; i++) s += cur[i] * L.w[i * L.nout + o];
      out[o] = li === ls.length - 1 ? s : Math.tanh(s);
    }
    acts.push(out);
    cur = out;
  }
  return acts;
}

export interface Bands {
  q10: Float32Array;
  q25: Float32Array;
  q75: Float32Array;
  q90: Float32Array;
}

export const BAND_BUCKETS = 5;

export interface TrainedModel {
  kind: "ridge" | "mlp";
  predictScaled(x: Float32Array): Float32Array;
  /**
   * Held-out residual quantiles, conditioned on how far into the path the
   * cursor sits. Pooling these across cursor positions is wrong: a cursor at
   * 20% has four times the remaining horizon of one at 80%, so its scaled
   * residuals are far larger and would swamp the pool, drawing a band several
   * times too wide late in the session.
   */
  bands(tFrac: number): Bands;
}

export interface TrainReport {
  trainRows: number;
  valRows: number;
  trainSessions: number;
  /** Last training date — anything on or after this is out of sample. */
  boundary: string;
  /** Epoch training actually stopped at. */
  stoppedAt: number;
  ridgeValR2: number;
  mlpValR2: number;
  ridgeValCorr: number;
  mlpValCorr: number;
  epochs: number;
}

function r2AndCorr(pred: Float32Array[], act: Float32Array[]) {
  let ssRes = 0,
    ssTot = 0,
    n = 0,
    mean = 0;
  for (const a of act) for (let k = 0; k < HORIZONS; k++) (mean += a[k]), n++;
  mean /= Math.max(1, n);
  let pm = 0,
    am = 0;
  for (let i = 0; i < act.length; i++)
    for (let k = 0; k < HORIZONS; k++) {
      pm += pred[i][k];
      am += act[i][k];
    }
  pm /= Math.max(1, n);
  am /= Math.max(1, n);
  let cov = 0,
    vp = 0,
    va = 0;
  for (let i = 0; i < act.length; i++)
    for (let k = 0; k < HORIZONS; k++) {
      const d = act[i][k] - pred[i][k];
      ssRes += d * d;
      ssTot += (act[i][k] - mean) * (act[i][k] - mean);
      const dp = pred[i][k] - pm;
      const da = act[i][k] - am;
      cov += dp * da;
      vp += dp * dp;
      va += da * da;
    }
  return {
    r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
    corr: vp > 0 && va > 0 ? cov / Math.sqrt(vp * va) : 0,
  };
}

function quantile(sorted: number[], q: number) {
  if (sorted.length === 0) return 0;
  const p = (sorted.length - 1) * q;
  const lo = Math.floor(p);
  const hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}

function quantilesOver(
  pred: Float32Array[],
  act: Float32Array[],
  pick: number[]
): Bands {
  const mk = () => new Float32Array(HORIZONS);
  const q10 = mk(),
    q25 = mk(),
    q75 = mk(),
    q90 = mk();
  for (let k = 0; k < HORIZONS; k++) {
    const res: number[] = [];
    for (const i of pick) res.push(act[i][k] - pred[i][k]);
    res.sort((a, b) => a - b);
    q10[k] = quantile(res, 0.1);
    q25[k] = quantile(res, 0.25);
    q75[k] = quantile(res, 0.75);
    q90[k] = quantile(res, 0.9);
  }
  return { q10, q25, q75, q90 };
}

export function bucketOf(tFrac: number) {
  return Math.max(0, Math.min(BAND_BUCKETS - 1, Math.floor(tFrac * BAND_BUCKETS)));
}

/** Per-bucket bands, with the pooled set as fallback for thin buckets. */
function makeBander(
  pred: Float32Array[],
  act: Float32Array[],
  tFracs: number[]
): (tFrac: number) => Bands {
  const all = pred.map((_, i) => i);
  const pooled = quantilesOver(pred, act, all);
  const buckets: (Bands | null)[] = [];
  for (let b = 0; b < BAND_BUCKETS; b++) {
    const pick = all.filter((i) => bucketOf(tFracs[i]) === b);
    buckets.push(pick.length >= 40 ? quantilesOver(pred, act, pick) : null);
  }
  return (tFrac: number) => buckets[bucketOf(tFrac)] ?? pooled;
}

export interface Trained {
  ridge: TrainedModel;
  mlp: TrainedModel;
  scaler: Scaler;
  options: ModelOptions;
  report: TrainReport;
}

/**
 * Fit on `corpus`, which the caller must have already truncated to sessions
 * strictly before the target. The last 20% of sessions become a validation
 * slice for early stopping, residual bands, and the honest metrics -- the
 * optimizer never sees it.
 */
export function trainModel(
  corpus: NormDay[],
  optsIn: Partial<ModelOptions> = {}
): Trained | null {
  const opts = { ...DEFAULT_OPTIONS, ...optsIn };
  if (corpus.length < 60) return null;

  const cut = Math.floor(corpus.length * 0.8);
  const trainDays = corpus.slice(0, cut);
  const valDays = corpus.slice(cut);

  const trainRows = buildSamples(trainDays, opts);
  const valRows = buildSamples(valDays, opts);
  if (trainRows.length < 100 || valRows.length < 20) return null;

  const scaler = fitScaler(trainRows);
  const Xtr = trainRows.map((r) => applyScaler(scaler, r.x));
  const Xva = valRows.map((r) => applyScaler(scaler, r.x));
  const Ytr = trainRows.map((r) => r.y);
  const Yva = valRows.map((r) => r.y);

  // ---- ridge
  const rw = fitRidge(trainRows, scaler, opts.l2);
  const ridgeVal = Xva.map((x) => ridgePredict(rw, x));
  const ridgeStats = r2AndCorr(ridgeVal, Yva);
  const vaFrac = valRows.map((r) => r.x[FEAT_POINTS + 2]);
  const rq = makeBander(ridgeVal, Yva, vaFrac);

  // ---- mlp, Adam with early stopping on validation MSE
  const rng = mulberry32(opts.seed + 1);
  const dims = [FEAT_DIM, ...opts.hidden, HORIZONS];
  const ls = initLayers(dims, rng);
  let best: Layer[] | null = null;
  let bestMse = Infinity;
  let bestEpoch = 0;
  let stale = 0;
  let stopped = opts.epochs;
  const b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8;
  let step = 0;

  const order = Xtr.map((_, i) => i);
  const batch = 64;

  for (let ep = 1; ep <= opts.epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let s = 0; s < order.length; s += batch) {
      const idx = order.slice(s, s + batch);
      const gW = ls.map((L) => new Float32Array(L.w.length));
      const gB = ls.map((L) => new Float32Array(L.b.length));

      for (const i of idx) {
        const acts = forward(ls, Xtr[i]);
        let delta = new Float32Array(HORIZONS);
        const out = acts[acts.length - 1];
        for (let k = 0; k < HORIZONS; k++)
          delta[k] = (2 * (out[k] - Ytr[i][k])) / HORIZONS;

        for (let li = ls.length - 1; li >= 0; li--) {
          const L = ls[li];
          const inp = acts[li];
          for (let o = 0; o < L.nout; o++) {
            const d = delta[o];
            if (d === 0) continue;
            gB[li][o] += d;
            for (let a = 0; a < L.nin; a++) gW[li][a * L.nout + o] += inp[a] * d;
          }
          if (li > 0) {
            const prev = new Float32Array(L.nin);
            for (let a = 0; a < L.nin; a++) {
              let s2 = 0;
              for (let o = 0; o < L.nout; o++) s2 += L.w[a * L.nout + o] * delta[o];
              prev[a] = s2 * (1 - inp[a] * inp[a]); // tanh'
            }
            delta = prev;
          }
        }
      }

      step++;
      const bc1 = 1 - Math.pow(b1, step);
      const bc2 = 1 - Math.pow(b2, step);
      for (let li = 0; li < ls.length; li++) {
        const L = ls[li];
        const inv = 1 / idx.length;
        for (let j = 0; j < L.w.length; j++) {
          const g = gW[li][j] * inv + opts.l2 * L.w[j];
          L.mw[j] = b1 * L.mw[j] + (1 - b1) * g;
          L.vw[j] = b2 * L.vw[j] + (1 - b2) * g * g;
          L.w[j] -= (opts.lr * (L.mw[j] / bc1)) / (Math.sqrt(L.vw[j] / bc2) + eps);
        }
        for (let j = 0; j < L.b.length; j++) {
          const g = gB[li][j] * inv;
          L.mb[j] = b1 * L.mb[j] + (1 - b1) * g;
          L.vb[j] = b2 * L.vb[j] + (1 - b2) * g * g;
          L.b[j] -= (opts.lr * (L.mb[j] / bc1)) / (Math.sqrt(L.vb[j] / bc2) + eps);
        }
      }
    }

    if (ep % 5 === 0 || ep === opts.epochs) {
      let mse = 0;
      for (let i = 0; i < Xva.length; i++) {
        const p = forward(ls, Xva[i])[ls.length];
        for (let k = 0; k < HORIZONS; k++) {
          const d = p[k] - Yva[i][k];
          mse += d * d;
        }
      }
      mse /= Xva.length * HORIZONS;
      if (mse < bestMse - 1e-9) {
        bestMse = mse;
        bestEpoch = ep;
        stale = 0;
        best = ls.map((L) => ({
          ...L,
          w: Float32Array.from(L.w),
          b: Float32Array.from(L.b),
        }));
      } else if (++stale >= opts.patience) {
        stopped = ep;
        break;
      }
    }
  }

  const useLs = best ?? ls;
  const mlpVal = Xva.map((x) => forward(useLs, x)[useLs.length]);
  const mlpStats = r2AndCorr(mlpVal, Yva);
  const mq = makeBander(mlpVal, Yva, vaFrac);

  const uniqTrain = new Set(trainRows.map((r) => r.date)).size;

  return {
    scaler,
    options: opts,
    ridge: {
      kind: "ridge",
      predictScaled: (x) => ridgePredict(rw, x),
      bands: rq,
    },
    mlp: {
      kind: "mlp",
      predictScaled: (x) => forward(useLs, x)[useLs.length],
      bands: mq,
    },
    report: {
      trainRows: trainRows.length,
      valRows: valRows.length,
      trainSessions: uniqTrain,
      boundary: corpus[corpus.length - 1].date,
      ridgeValR2: ridgeStats.r2,
      mlpValR2: mlpStats.r2,
      ridgeValCorr: ridgeStats.corr,
      mlpValCorr: mlpStats.corr,
      epochs: bestEpoch,
      stoppedAt: stopped,
    },
  };
}

// -------------------------------------------------------------- projection

/**
 * Turn a prediction into the same Projection shape the chart already draws,
 * so the model and the analog fan render through one code path.
 *
 * The band is the held-out residual spread, not a spread across ensemble
 * members -- it says "predictions like this one have missed by this much
 * before", which is the only uncertainty claim the data supports.
 */
export function modelProjection(
  trained: Trained,
  model: TrainedModel,
  day: NormDay,
  t: number
): Projection | null {
  const f = makeFeatures(day, t, trained.options.windowBars, trained.options.anchored);
  if (!f) return null;

  const n = day.r.length;
  const anchor = day.closes[t];
  const xs = applyScaler(trained.scaler, f.x);
  const yhat = model.predictScaled(xs);
  const band = model.bands(t / (n - 1));

  const horizon = n - t;
  const idxs = [0];
  const vals = [0];
  const lo10 = [0],
    lo25 = [0],
    hi75 = [0],
    hi90 = [0];
  for (let k = 0; k < HORIZONS; k++) {
    const idx = Math.round(t + ((n - 1 - t) * (k + 1)) / HORIZONS) - t;
    idxs.push(idx);
    vals.push(yhat[k] * f.scale);
    lo10.push((yhat[k] + band.q10[k]) * f.scale);
    lo25.push((yhat[k] + band.q25[k]) * f.scale);
    hi75.push((yhat[k] + band.q75[k]) * f.scale);
    hi90.push((yhat[k] + band.q90[k]) * f.scale);
  }

  const fill = (pts: number[]) => {
    const out = new Float32Array(horizon);
    let seg = 0;
    for (let j = 0; j < horizon; j++) {
      while (seg < idxs.length - 2 && j > idxs[seg + 1]) seg++;
      const a = idxs[seg],
        b = idxs[seg + 1];
      const w = b === a ? 0 : (j - a) / (b - a);
      out[j] = anchor * Math.exp(pts[seg] + (pts[seg + 1] - pts[seg]) * w);
    }
    return out;
  };

  const p50 = fill(vals);
  const one = new Float32Array(1);
  one[0] = anchor;

  return {
    from: t,
    lines: [],
    mean: p50,
    p10: fill(lo10),
    p25: fill(lo25),
    p50,
    p75: fill(hi75),
    p90: fill(hi90),
    backFrom: t,
    backLines: [],
    backP10: one,
    backP25: one,
    backP50: one,
    backP75: one,
    backP90: one,
  };
}

// ------------------------------------------------------------ walk-forward

export interface ModelEval {
  n: number;
  hitRate: number;
  baselineLong: number;
  baselineMomentum: number;
  meanPathCorr: number;
}

/**
 * Score a trained model on sessions after the training boundary, using the
 * same definitions as backtest() in engine.ts so the two are comparable.
 */
export function evalModel(
  trained: Trained,
  model: TrainedModel,
  testDays: NormDay[],
  t: number
): ModelEval {
  let hits = 0,
    long = 0,
    mom = 0,
    n = 0,
    corrSum = 0,
    corrN = 0;

  for (const day of testDays) {
    const nb = day.r.length;
    if (t >= nb - 2 || day.lastReal < nb - 2) continue;
    const proj = modelProjection(trained, model, day, t);
    if (!proj) continue;

    const anchor = day.closes[t];
    const actualEnd = day.closes[nb - 1];
    const predEnd = proj.p50[proj.p50.length - 1];

    const actualDir = Math.sign(actualEnd - anchor);
    if (actualDir === 0) continue;
    n++;
    if (Math.sign(predEnd - anchor) === actualDir) hits++;
    if (actualDir > 0) long++;
    if (Math.sign(day.closes[t] - day.closes[0]) === actualDir) mom++;

    // Correlation of predicted vs realized forward path.
    const h = nb - t;
    let pm = 0,
      am = 0;
    for (let j = 0; j < h; j++) {
      pm += proj.p50[j];
      am += day.closes[t + j];
    }
    pm /= h;
    am /= h;
    let cov = 0,
      vp = 0,
      va = 0;
    for (let j = 0; j < h; j++) {
      const dp = proj.p50[j] - pm;
      const da = day.closes[t + j] - am;
      cov += dp * da;
      vp += dp * dp;
      va += da * da;
    }
    if (vp > 0 && va > 0) {
      corrSum += cov / Math.sqrt(vp * va);
      corrN++;
    }
  }

  return {
    n,
    hitRate: n ? hits / n : 0,
    baselineLong: n ? long / n : 0,
    baselineMomentum: n ? mom / n : 0,
    meanPathCorr: corrN ? corrSum / corrN : 0,
  };
}
