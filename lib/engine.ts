/**
 * Analog-day engine.
 *
 * Everything here is pure and runs client-side. The whole corpus is a few MB of
 * Float32, and a full scan is a few hundred thousand flops, so there is no
 * reason to put any of this behind a network call.
 */

/** ES session: 18:00 ET (prev day) -> 17:00 ET, 23h, 1-minute bars. */
export const SESSION_BARS = 1380;

/** Minutes from session anchor (18:00 ET) to the 09:30 ET cash open. */
export const RTH_OPEN_BAR = 930;
/** Minutes from session anchor to the 16:00 ET cash close. */
export const RTH_CLOSE_BAR = 1320;

export interface DayPath {
  /** Settlement date, YYYY-MM-DD. */
  date: string;
  /** Close of each 1-minute bar, indexed by minutes since session anchor. */
  closes: Float32Array;
}

export interface NormDay {
  date: string;
  /** Gap-filled closes. */
  closes: Float32Array;
  /** log(close[i] / close[0]) — the session path in return space. */
  r: Float32Array;
  /** Index of the last bar with real (non-extrapolated) data. */
  lastReal: number;
}

/**
 * Fill gaps and convert to log-return-from-anchor.
 *
 * Matching on raw price levels is the single most common way to break this kind
 * of tool: within-day price is dominated by level and drift, so unrelated days
 * correlate at 0.99 purely because both went up. Returns from the session anchor
 * remove that.
 */
export function normalize(day: DayPath): NormDay {
  const n = day.closes.length;
  const c = new Float32Array(n);

  let last = NaN;
  let lastReal = -1;
  for (let i = 0; i < n; i++) {
    const v = day.closes[i];
    if (Number.isFinite(v) && v > 0) {
      last = v;
      lastReal = i;
    }
    c[i] = last;
  }

  // Back-fill any leading gap with the first real print.
  let first = NaN;
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(c[i])) {
      first = c[i];
      break;
    }
  }
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(c[i])) break;
    c[i] = first;
  }

  const r = new Float32Array(n);
  const base = c[0];
  for (let i = 0; i < n; i++) r[i] = Math.log(c[i] / base);

  return { date: day.date, closes: c, r, lastReal };
}

export type Metric = "pearson" | "euclid";

export interface ScanOptions {
  /** Current bar index — "now". */
  t: number;
  /** Lookback length in minutes. Slots are minutes-since-anchor, so this is
   *  timeframe-independent. Ignored when `anchored` is true. */
  windowMinutes: number;
  /** Match the whole session-to-date (0..t) instead of a trailing window. */
  anchored: boolean;
  topK: number;
  metric: Metric;
  /** Exclude candidate days within this many calendar days of the target. */
  excludeNearDays?: number;
}

export interface Match {
  date: string;
  dayIndex: number;
  /** Pearson correlation over the window. Signed. */
  corr: number;
  /** Ranking score. Higher is a better match. */
  score: number;
  /** True when this is an inverted analog. */
  inverse: boolean;
  /** targetVol / candidateVol, clamped. Used to rescale the projection. */
  volRatio: number;
}

export interface ScanResult {
  lo: number;
  hi: number;
  positive: Match[];
  negative: Match[];
  /** Best |corr| found — useful as a noise diagnostic. */
  bestAbsCorr: number;
}

interface WindowStats {
  /** Re-baselined path: r[i] - r[lo]. */
  path: Float32Array;
  mean: number;
  sd: number;
  /** Stdev of bar-to-bar increments — the window's realized vol. */
  vol: number;
}

function windowStats(r: Float32Array, lo: number, hi: number): WindowStats {
  const w = hi - lo + 1;
  const path = new Float32Array(w);
  const base = r[lo];
  let sum = 0;
  for (let i = 0; i < w; i++) {
    const v = r[lo + i] - base;
    path[i] = v;
    sum += v;
  }
  const mean = sum / w;

  let ss = 0;
  for (let i = 0; i < w; i++) {
    const d = path[i] - mean;
    ss += d * d;
  }
  const sd = Math.sqrt(ss / w);

  let vs = 0;
  let vm = 0;
  for (let i = 1; i < w; i++) vm += path[i] - path[i - 1];
  vm /= Math.max(1, w - 1);
  for (let i = 1; i < w; i++) {
    const d = path[i] - path[i - 1] - vm;
    vs += d * d;
  }
  const vol = Math.sqrt(vs / Math.max(1, w - 1));

  return { path, mean, sd, vol };
}

function daysBetween(a: string, b: string): number {
  return Math.abs(Date.parse(a) - Date.parse(b)) / 86400000;
}

/**
 * Compare the target day's window against every candidate day at the same
 * clock offsets, and return the best positive and best inverse analogs.
 *
 * The inverse list is not a second pass — it is just the other end of the same
 * sorted correlation vector, so it costs nothing. It also doubles as a
 * diagnostic: if the best positive and best inverse are both near 1 in
 * magnitude, the window is too short to be distinguishing anything.
 */
export function scan(
  target: NormDay,
  corpus: NormDay[],
  opts: ScanOptions
): ScanResult {
  const { t, anchored, topK, metric } = opts;
  const lo = anchored ? 0 : Math.max(0, t - opts.windowMinutes + 1);
  const hi = t;
  const w = hi - lo + 1;

  const tgt = windowStats(target.r, lo, hi);
  const results: Match[] = [];

  for (let d = 0; d < corpus.length; d++) {
    const cand = corpus[d];
    if (cand.date === target.date) continue;
    if (cand.lastReal < hi) continue;
    if (
      opts.excludeNearDays &&
      daysBetween(cand.date, target.date) <= opts.excludeNearDays
    ) {
      continue;
    }

    const cs = windowStats(cand.r, lo, hi);

    // Pearson over the window.
    let cov = 0;
    for (let i = 0; i < w; i++) {
      cov += (tgt.path[i] - tgt.mean) * (cs.path[i] - cs.mean);
    }
    cov /= w;
    const denom = tgt.sd * cs.sd;
    const corr = denom > 1e-12 ? cov / denom : 0;

    let score: number;
    let inverse: boolean;

    if (metric === "pearson") {
      score = Math.abs(corr);
      inverse = corr < 0;
    } else {
      // Euclidean on vol-normalized paths. Unlike Pearson this respects
      // magnitude, so a 0.1% day does not match a 3% day at score 1.0.
      const ts = tgt.vol > 1e-12 ? 1 / tgt.vol : 0;
      const csn = cs.vol > 1e-12 ? 1 / cs.vol : 0;
      let dPos = 0;
      let dNeg = 0;
      for (let i = 0; i < w; i++) {
        const a = tgt.path[i] * ts;
        const b = cs.path[i] * csn;
        dPos += (a - b) * (a - b);
        dNeg += (a + b) * (a + b);
      }
      inverse = dNeg < dPos;
      const dist = Math.sqrt(Math.min(dPos, dNeg) / w);
      score = 1 / (1 + dist);
    }

    const volRatio = cs.vol > 1e-12 ? clamp(tgt.vol / cs.vol, 0.25, 4) : 1;
    results.push({
      date: cand.date,
      dayIndex: d,
      corr,
      score,
      inverse,
      volRatio,
    });
  }

  const positive = results
    .filter((m) => !m.inverse)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  const negative = results
    .filter((m) => m.inverse)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  let bestAbsCorr = 0;
  for (const m of results) bestAbsCorr = Math.max(bestAbsCorr, Math.abs(m.corr));

  return { lo, hi, positive, negative, bestAbsCorr };
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

export interface Projection {
  /** Bar index the projection starts from (== t). */
  from: number;
  /** Price paths for bars t..SESSION_BARS-1, one per analog. */
  lines: { date: string; corr: number; inverse: boolean; path: Float32Array }[];
  /** Score-weighted ensemble mean, in price. */
  mean: Float32Array;
  p10: Float32Array;
  p25: Float32Array;
  p50: Float32Array;
  p75: Float32Array;
  p90: Float32Array;
}

/**
 * Project each analog's remaining session onto today's last price.
 *
 * Two things happen here that a naive implementation skips:
 *
 * 1. Vol rescaling. Pearson is scale-blind, so a matched day that ran 3x
 *    today's range would otherwise produce a projection 3x too wide. Each
 *    analog's forward path is scaled by targetVol/analogVol.
 * 2. A fan, not a line. The top single analog out of ~1250 candidates is
 *    almost always a multiple-comparisons artifact. The percentile bands show
 *    how much the top-K actually agree, which is the information you want.
 */
export function project(
  target: NormDay,
  corpus: NormDay[],
  matches: Match[],
  t: number,
  applyVolScaling = true
): Projection {
  const n = target.r.length;
  const horizon = n - t;
  const anchorPrice = target.closes[t];

  const lines = matches.map((m) => {
    const cand = corpus[m.dayIndex];
    const base = cand.r[t];
    const sign = m.inverse ? -1 : 1;
    const s = (applyVolScaling ? m.volRatio : 1) * sign;
    const path = new Float32Array(horizon);
    for (let j = 0; j < horizon; j++) {
      path[j] = anchorPrice * Math.exp(s * (cand.r[t + j] - base));
    }
    return { date: m.date, corr: m.corr, inverse: m.inverse, path };
  });

  const mk = () => new Float32Array(horizon);
  const mean = mk();
  const p10 = mk();
  const p25 = mk();
  const p50 = mk();
  const p75 = mk();
  const p90 = mk();

  if (lines.length === 0) {
    return { from: t, lines, mean, p10, p25, p50, p75, p90 };
  }

  const weights = matches.map((m) => Math.max(1e-6, m.score));
  const wsum = weights.reduce((a, b) => a + b, 0);
  const col = new Float64Array(lines.length);

  for (let j = 0; j < horizon; j++) {
    let acc = 0;
    for (let k = 0; k < lines.length; k++) {
      col[k] = lines[k].path[j];
      acc += lines[k].path[j] * weights[k];
    }
    mean[j] = acc / wsum;
    const sorted = Array.from(col).sort((a, b) => a - b);
    p10[j] = quantile(sorted, 0.1);
    p25[j] = quantile(sorted, 0.25);
    p50[j] = quantile(sorted, 0.5);
    p75[j] = quantile(sorted, 0.75);
    p90[j] = quantile(sorted, 0.9);
  }

  return { from: t, lines, mean, p10, p25, p50, p75, p90 };
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export interface BacktestResult {
  n: number;
  /** Fraction of days where the ensemble called the sign of the remaining move. */
  hitRate: number;
  /** Always-long baseline over the same days and horizon. */
  baselineLong: number;
  /** "Rest of session continues the session-to-date direction" baseline. */
  baselineMomentum: number;
  /** Mean |corr| of the top match — high values here mean overfitting. */
  meanBestCorr: number;
  /** Mean correlation between projected and realized forward path. */
  meanPathCorr: number;
  samples: {
    date: string;
    predicted: number;
    actual: number;
    bestCorr: number;
    hit: boolean;
  }[];
}

/**
 * Walk-forward test. For each test day, only days strictly earlier in the
 * corpus are eligible as analogs, so there is no lookahead.
 *
 * Read the output honestly: if hitRate does not clear both baselines by a
 * meaningful margin, the projection is decoration.
 */
export function backtest(
  corpus: NormDay[],
  opts: ScanOptions & { startIndex?: number; maxDays?: number }
): BacktestResult {
  const start = opts.startIndex ?? Math.floor(corpus.length * 0.5);
  const t = opts.t;
  const n = corpus[0].r.length;
  const samples: BacktestResult["samples"] = [];

  let hits = 0;
  let baseLong = 0;
  let baseMom = 0;
  let corrSum = 0;
  let pathCorrSum = 0;
  let pathCorrN = 0;

  const end = Math.min(corpus.length, start + (opts.maxDays ?? 250));

  for (let i = start; i < end; i++) {
    const test = corpus[i];
    if (test.lastReal < n - 1) continue;

    const history = corpus.slice(0, i);
    if (history.length < 50) continue;

    const res = scan(test, history, opts);
    const pool = [...res.positive, ...res.negative]
      .sort((a, b) => b.score - a.score)
      .slice(0, opts.topK);
    if (pool.length === 0) continue;

    const proj = project(test, history, pool, t);
    const predicted = proj.mean[proj.mean.length - 1] - test.closes[t];
    const actual = test.closes[n - 1] - test.closes[t];
    const sessionSoFar = test.closes[t] - test.closes[0];

    const hit = Math.sign(predicted) === Math.sign(actual) && actual !== 0;
    if (hit) hits++;
    if (actual > 0) baseLong++;
    if (Math.sign(sessionSoFar) === Math.sign(actual) && actual !== 0) baseMom++;
    corrSum += Math.abs(pool[0].corr);

    // Shape agreement over the forward path, not just the terminal sign.
    const h = n - t;
    if (h > 5) {
      const a: number[] = [];
      const b: number[] = [];
      for (let j = 0; j < h; j++) {
        a.push(proj.mean[j] - test.closes[t]);
        b.push(test.closes[t + j] - test.closes[t]);
      }
      pathCorrSum += pearson(a, b);
      pathCorrN++;
    }

    samples.push({
      date: test.date,
      predicted,
      actual,
      bestCorr: pool[0].corr,
      hit,
    });
  }

  const m = samples.length || 1;
  return {
    n: samples.length,
    hitRate: hits / m,
    baselineLong: baseLong / m,
    baselineMomentum: baseMom / m,
    meanBestCorr: corrSum / m,
    meanPathCorr: pathCorrN ? pathCorrSum / pathCorrN : 0,
    samples,
  };
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const d = Math.sqrt(va * vb);
  return d > 1e-12 ? cov / d : 0;
}
