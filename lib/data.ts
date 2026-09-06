import { DayPath, SESSION_BARS, RTH_OPEN_BAR, RTH_CLOSE_BAR } from "./engine";

/** Deterministic PRNG so demo data is stable across reloads. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Intraday volatility shape for ES: quiet Asian/European overnight, a sharp
 * step up at the 09:30 ET cash open, elevated into the close, dead after.
 */
function volShape(i: number): number {
  if (i < RTH_OPEN_BAR - 60) return 0.35;
  if (i < RTH_OPEN_BAR) return 0.7;
  const into = i - RTH_OPEN_BAR;
  if (i < RTH_CLOSE_BAR) {
    const openBurst = 1.6 * Math.exp(-into / 45);
    const closeRamp = 0.5 * Math.exp(-(RTH_CLOSE_BAR - i) / 60);
    return 0.9 + openBurst + closeRamp;
  }
  return 0.3;
}

/**
 * Generate a corpus of driftless random-walk sessions.
 *
 * These are deliberately structureless. If the tool appears to find strong
 * analogs and confident projections on this data, that is the multiple-
 * comparisons artifact showing itself, not a bug — which is exactly why the
 * demo ships with it.
 */
export function syntheticCorpus(days: number, seed = 42): DayPath[] {
  const rng = mulberry32(seed);
  const out: DayPath[] = [];
  let price = 5200;

  const cursor = new Date(Date.UTC(2021, 0, 4));
  while (out.length < days) {
    const dow = cursor.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      const closes = new Float32Array(SESSION_BARS);
      const dayVol = 0.55 + rng() * 1.1;
      let p = price;
      for (let i = 0; i < SESSION_BARS; i++) {
        const sigma = dayVol * volShape(i) * 0.00035;
        // Itô correction — without it, exp(sum of gaussians) drifts upward and
        // 1.7M steps compounds into a nonsense price level.
        p *= Math.exp(sigma * gauss(rng) - (sigma * sigma) / 2);
        closes[i] = p;
      }
      price = p;
      out.push({ date: cursor.toISOString().slice(0, 10), closes });
    }
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return out;
}

/**
 * Pull real bars through the /api/bars proxy and parse them into sessions.
 *
 * The proxy keeps DATABENTO_API_KEY server-side, so nothing sensitive reaches
 * the browser. It serves Databento's *historical* endpoint, which lags roughly
 * 24 hours — fine for corpus building and replay, useless for a live tail.
 *
 * Throws with the server's own message on failure, including the 501 you get
 * when the key has not been set on the deployment.
 */
export async function fetchBars(
  start: string,
  end: string,
  symbol = "ES.c.0"
): Promise<DayPath[]> {
  const q = new URLSearchParams({ start, symbol });
  if (end) q.set("end", end);

  const r = await fetch(`/api/bars?${q}`);
  if (!r.ok) {
    let msg = `Request failed (${r.status}).`;
    try {
      const j = await r.json();
      if (j?.error) msg = j.fix ? `${j.error} ${j.fix}` : String(j.error);
    } catch {
      // Non-JSON error body; the status line is all we have.
    }
    throw new Error(msg);
  }

  const csv = await r.text();
  const days = parseBars(csv);
  if (days.length === 0) {
    throw new Error("No sessions parsed from that range.");
  }
  return days;
}

/**
 * Parse 1-minute bars into session-aligned days.
 *
 * Expects `timestamp,close` or Databento's `ts_event,...,close` shape. The
 * timestamp must be ISO 8601 with a UTC offset — bar alignment is done in ET
 * so that DST shifts do not smear the session anchor by an hour twice a year.
 */
export function parseBars(csv: string): DayPath[] {
  const lines = csv.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const tsIdx = header.findIndex((h) =>
    ["ts_event", "timestamp", "time", "datetime", "date"].includes(h)
  );
  const closeIdx = header.findIndex((h) => ["close", "c", "last"].includes(h));
  if (tsIdx < 0 || closeIdx < 0) {
    throw new Error(
      "Need a timestamp column (ts_event/timestamp/time) and a close column."
    );
  }

  const byDay = new Map<string, Float32Array>();

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length <= Math.max(tsIdx, closeIdx)) continue;
    const ts = parseTimestamp(parts[tsIdx].trim());
    const close = parseFloat(parts[closeIdx]);
    if (!Number.isFinite(ts) || !Number.isFinite(close)) continue;

    const slot = sessionSlot(ts);
    if (!slot) continue;
    let arr = byDay.get(slot.date);
    if (!arr) {
      arr = new Float32Array(SESSION_BARS).fill(NaN);
      byDay.set(slot.date, arr);
    }
    arr[slot.bar] = close;
  }

  return Array.from(byDay.entries())
    .map(([date, closes]) => ({ date, closes }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * Accept ISO 8601 or a bare epoch integer.
 *
 * Databento emits ts_event as nanoseconds since epoch in its raw CSV, but as an
 * ISO string via the Python client's to_df().to_csv(). Both turn up in practice,
 * so infer from magnitude rather than guessing.
 */
function parseTimestamp(raw: string): number {
  if (/^\d+$/.test(raw)) {
    const n = Number(raw);
    if (n > 1e17) return n / 1e6; // nanoseconds
    if (n > 1e14) return n / 1e3; // microseconds
    if (n > 1e11) return n; // milliseconds
    return n * 1e3; // seconds
  }
  // A timestamp with no zone marker is ambiguous; assume UTC, as every vendor
  // in this space emits UTC.
  const hasZone = /[zZ]|[+-]\d{2}:?\d{2}$/.test(raw);
  return Date.parse(hasZone ? raw : `${raw.replace(" ", "T")}Z`);
}

/**
 * Map a UTC instant to (settlement date, minutes since 18:00 ET).
 *
 * The 17:00–18:00 ET maintenance break is excluded. Uses Intl rather than a
 * fixed offset so DST is handled correctly.
 */
const ET = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export function sessionSlot(
  tsMs: number
): { date: string; bar: number } | null {
  const parts = ET.formatToParts(new Date(tsMs));
  const get = (t: string) => parts.find((p) => p.type === t)!.value;
  const y = +get("year");
  const mo = +get("month");
  const d = +get("day");
  let h = +get("hour");
  const mi = +get("minute");
  if (h === 24) h = 0;

  const minutes = h * 60 + mi;
  const ANCHOR = 18 * 60; // 18:00 ET
  const BREAK = 17 * 60; // 17:00 ET

  if (minutes >= BREAK && minutes < ANCHOR) return null; // maintenance

  let bar: number;
  const local = new Date(Date.UTC(y, mo - 1, d));

  if (minutes >= ANCHOR) {
    // Evening session belongs to the *next* settlement date.
    bar = minutes - ANCHOR;
    local.setUTCDate(local.getUTCDate() + 1);
  } else {
    bar = minutes + (24 * 60 - ANCHOR);
  }

  if (bar < 0 || bar >= SESSION_BARS) return null;
  return { date: local.toISOString().slice(0, 10), bar };
}
