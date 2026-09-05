/**
 * Does the match hold up forward?
 *
 * Matches at a chosen bar, then measures how well the projection tracks what
 * actually happened over the following horizons. Run it on synthetic controls
 * to calibrate, then on real bars:
 *
 *     npx tsx scripts/persistence.ts
 *     npx tsx scripts/persistence.ts --csv es_1m.csv
 *
 * Two synthetic controls run by default:
 *
 *   NEGATIVE (snr 0.0)  driftless random walks, no structure at all. Forward
 *                       correlation must come out near zero. If it does not,
 *                       something is leaking lookahead.
 *
 *   POSITIVE (snr > 0)  each session is built from one of a few recurring
 *                       archetype shapes plus noise. Structure genuinely is
 *                       there and repeats. If the method cannot find it here,
 *                       a null result on real data means the tool is broken,
 *                       not that the market lacks analogs.
 *
 * Without the positive control a null tells you nothing.
 */

import { normalize, scan, project, NormDay, DayPath, SESSION_BARS } from "../lib/engine";
import { syntheticCorpus, parseBars } from "../lib/data";
import * as fs from "fs";

const MATCH_BAR = 1080; // 12:00 ET
const WINDOW = 120;
const TOPK = 25;
const HORIZONS = [30, 60, 120, 240]; // +30m, +1h, +2h, +4h (16:00 ET)

// --------------------------------------------------------------- generators
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
function gauss(rng: () => number) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Smooth low-frequency path — a plausible "session shape". */
function archetype(rng: () => number): Float32Array {
  const p = new Float32Array(SESSION_BARS);
  const comps = 5;
  const amp: number[] = [];
  const phase: number[] = [];
  const freq: number[] = [];
  for (let k = 0; k < comps; k++) {
    amp.push((rng() - 0.5) * 2);
    phase.push(rng() * Math.PI * 2);
    freq.push(0.5 + k);
  }
  for (let i = 0; i < SESSION_BARS; i++) {
    const x = i / SESSION_BARS;
    let v = 0;
    for (let k = 0; k < comps; k++) {
      v += (amp[k] / freq[k]) * Math.sin(Math.PI * freq[k] * x + phase[k]);
    }
    p[i] = v;
  }
  // Normalise to unit terminal-ish scale.
  let mx = 0;
  for (let i = 0; i < SESSION_BARS; i++) mx = Math.max(mx, Math.abs(p[i]));
  if (mx > 0) for (let i = 0; i < SESSION_BARS; i++) p[i] /= mx;
  return p;
}

/**
 * Blend a recurring archetype with an independent random walk.
 * snr = 0 is pure noise; snr = 1 is pure repeating shape.
 */
function structuredCorpus(days: number, snr: number, nArch: number, seed: number): DayPath[] {
  const rng = mulberry32(seed);
  const arch: Float32Array[] = [];
  for (let k = 0; k < nArch; k++) arch.push(archetype(rng));

  const out: DayPath[] = [];
  const cursor = new Date(Date.UTC(2019, 0, 2));
  let price = 4000;

  while (out.length < days) {
    const dow = cursor.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      const a = arch[Math.floor(rng() * nArch)];
      const dayVol = 0.008 * (0.6 + rng() * 0.9);

      // Independent noise path in return space.
      const noise = new Float32Array(SESSION_BARS);
      let acc = 0;
      const step = 1 / Math.sqrt(SESSION_BARS);
      for (let i = 0; i < SESSION_BARS; i++) {
        acc += gauss(rng) * step;
        noise[i] = acc;
      }

      const closes = new Float32Array(SESSION_BARS);
      const w = Math.sqrt(Math.max(0, 1 - snr * snr));
      for (let i = 0; i < SESSION_BARS; i++) {
        const r = dayVol * (snr * a[i] + w * noise[i]);
        closes[i] = price * Math.exp(r);
      }
      price = closes[SESSION_BARS - 1];
      out.push({ date: cursor.toISOString().slice(0, 10), closes });
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return out;
}

// ------------------------------------------------------------------- stats
function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n < 3) return 0;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    cov += da * db; va += da * da; vb += db * db;
  }
  const d = Math.sqrt(va * vb);
  return d > 1e-12 ? cov / d : 0;
}
const mean = (x: number[]) => (x.length ? x.reduce((p, c) => p + c, 0) / x.length : 0);

interface Row {
  horizon: number;
  pathCorr: number;
  hitRate: number;
  n: number;
}

interface Out {
  rows: Row[];
  matchCorr: number;
  buckets: { lo: number; hi: number; n: number; fwd: number; hit: number }[];
}

function run(corpus: NormDay[], startFrac: number, maxDays: number): Out {
  const start = Math.max(60, Math.floor(corpus.length * startFrac));
  const end = Math.min(corpus.length, start + maxDays);

  const fwd: Record<number, number[]> = {};
  const hit: Record<number, number[]> = {};
  for (const h of HORIZONS) { fwd[h] = []; hit[h] = []; }
  const matchCorrs: number[] = [];
  const pairs: { m: number; f: number; h: number }[] = [];

  for (let i = start; i < end; i++) {
    const test = corpus[i];
    if (test.lastReal < MATCH_BAR + Math.max(...HORIZONS)) continue;
    const history = corpus.slice(0, i);
    if (history.length < 60) continue;

    const res = scan(test, history, {
      t: MATCH_BAR,
      windowMinutes: WINDOW,
      anchored: false,
      topK: TOPK,
      metric: "pearson",
      excludeNearDays: 3,
    });
    const pool = [...res.positive, ...res.negative]
      .sort((a, b) => b.score - a.score)
      .slice(0, TOPK);
    if (pool.length < 5) continue;

    const proj = project(test, history, pool, MATCH_BAR);
    const anchor = test.closes[MATCH_BAR];
    matchCorrs.push(Math.abs(pool[0].corr));

    for (const h of HORIZONS) {
      const a: number[] = [];
      const b: number[] = [];
      for (let j = 0; j <= h; j++) {
        a.push(proj.p50[j] - anchor);
        b.push(test.closes[MATCH_BAR + j] - anchor);
      }
      const c = pearson(a, b);
      fwd[h].push(c);

      const pred = proj.p50[h] - anchor;
      const act = test.closes[MATCH_BAR + h] - anchor;
      const isHit = Math.sign(pred) === Math.sign(act) && act !== 0;
      hit[h].push(isHit ? 1 : 0);

      if (h === 240) pairs.push({ m: Math.abs(pool[0].corr), f: c, h: isHit ? 1 : 0 });
    }
  }

  const rows: Row[] = HORIZONS.map((h) => ({
    horizon: h,
    pathCorr: mean(fwd[h]),
    hitRate: mean(hit[h]),
    n: fwd[h].length,
  }));

  // Does a *better* in-sample match buy you a better forward result?
  const edges = [0.8, 0.9, 0.95, 0.98, 1.01];
  const buckets = [];
  for (let k = 0; k < edges.length - 1; k++) {
    const sel = pairs.filter((p) => p.m >= edges[k] && p.m < edges[k + 1]);
    buckets.push({
      lo: edges[k],
      hi: edges[k + 1],
      n: sel.length,
      fwd: mean(sel.map((s) => s.f)),
      hit: mean(sel.map((s) => s.h)),
    });
  }

  return { rows, matchCorr: mean(matchCorrs), buckets };
}

function report(label: string, o: Out) {
  console.log(`\n${label}`);
  console.log(`  mean in-sample match |corr| at 12:00 ET   ${o.matchCorr.toFixed(3)}`);
  console.log("  horizon      fwd path corr   direction hit   n");
  for (const r of o.rows) {
    const hm = r.horizon >= 60 ? `+${r.horizon / 60}h` : `+${r.horizon}m`;
    const se = 50 / Math.sqrt(Math.max(1, r.n));
    console.log(
      `  ${hm.padEnd(10)}   ${r.pathCorr.toFixed(4).padStart(13)}   ` +
        `${(r.hitRate * 100).toFixed(1).padStart(8)}% ±${se.toFixed(1)}   ${r.n}`
    );
  }
  console.log("  by match quality (at +4h):");
  for (const b of o.buckets) {
    if (b.n < 5) continue;
    console.log(
      `    |corr| ${b.lo.toFixed(2)}-${b.hi.toFixed(2)}   n=${String(b.n).padStart(3)}` +
        `   fwd corr ${b.fwd.toFixed(4).padStart(8)}   hit ${(b.hit * 100).toFixed(1)}%`
    );
  }
}

// -------------------------------------------------------------------- main
const csvArg = process.argv.indexOf("--csv");
console.log(
  `Match at bar ${MATCH_BAR} (12:00 ET), ${WINDOW}-bar window, top ${TOPK} analogs.\n` +
    "Forward path correlation is between the ensemble median projection and\n" +
    "what actually happened. Walk-forward: candidates are strictly earlier days."
);

if (csvArg > -1) {
  const path = process.argv[csvArg + 1];
  const days = parseBars(fs.readFileSync(path, "utf8"));
  console.log(`\nLoaded ${days.length} sessions from ${path}`);
  if (days.length < 120) {
    console.log("Need at least ~120 sessions for this to mean anything.");
  } else {
    report(`REAL DATA — ${path}`, run(days.map(normalize), 0.4, 400));
  }
} else {
  report(
    "NEGATIVE CONTROL — driftless random walks (expect ~0)",
    run(syntheticCorpus(900, 7).map(normalize), 0.45, 200)
  );
  for (const snr of [0.3, 0.6, 0.9]) {
    report(
      `POSITIVE CONTROL — 6 recurring archetypes, snr ${snr.toFixed(1)}`,
      run(structuredCorpus(900, snr, 6, 11).map(normalize), 0.45, 200)
    );
  }
  console.log(
    "\nRe-run with --csv once you have real bars. The number to compare is\n" +
      "forward path correlation against the negative control, not against zero."
  );
}
