import {
  DayPath,
  SESSION_BARS,
  RTH_OPEN_BAR,
  RTH_CLOSE_BAR,
} from "./engine";
import type { Mark } from "../components/Chart";

/**
 * Weekly paths: five sessions laid end to end and treated as one series.
 *
 * The engine never assumed a 1380-bar session -- normalize() reads
 * closes.length, scan() windows over .r, project() takes its horizon from
 * target.r.length -- so a week is just a longer day as far as matching is
 * concerned. Nothing in engine.ts needed to change for this.
 *
 * Two framings:
 *
 *   24h  full session, 18:00 -> 17:00 ET, 1380 bars/day, 6900 per week.
 *        Includes every overnight, so a Sunday-evening gap or a Wednesday
 *        overnight reversal is inside the matched shape.
 *
 *   rth  cash hours only, 09:30 -> 16:00 ET, 390 bars/day, 1950 per week.
 *        Splices out the overnights entirely: Monday's 16:00 close sits
 *        directly against Tuesday's 09:30 open. Overnight gaps become
 *        single-bar jumps rather than paths.
 *
 * These answer different questions and are not comparable. A week that looks
 * like a clean trend in RTH may be a chop-fest that gapped its way there.
 */
export type WeekMode = "24h" | "rth";

export const RTH_BARS = RTH_CLOSE_BAR - RTH_OPEN_BAR;

export const DAY_BARS: Record<WeekMode, number> = {
  "24h": SESSION_BARS,
  rth: RTH_BARS,
};

export const WEEK_BARS: Record<WeekMode, number> = {
  "24h": SESSION_BARS * 5,
  rth: RTH_BARS * 5,
};

export const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"];

export interface WeekBuild {
  weeks: DayPath[];
  /** Weeks seen but discarded because a session was missing. */
  incomplete: number;
}

/** Monday of the ISO week containing a YYYY-MM-DD settlement date. */
function mondayOf(date: string): string | null {
  const d = new Date(`${date}T00:00:00Z`);
  const dow = d.getUTCDay(); // 0 Sun .. 6 Sat
  if (dow < 1 || dow > 5) return null; // weekend settlement: not a thing
  d.setUTCDate(d.getUTCDate() - (dow - 1));
  return d.toISOString().slice(0, 10);
}

function weekdayIndex(date: string): number {
  return new Date(`${date}T00:00:00Z`).getUTCDay() - 1; // Mon -> 0
}

/**
 * Assemble complete Mon-Fri weeks from a list of sessions.
 *
 * Only complete weeks survive. A holiday week could be kept by leaving the
 * missing day as a gap, but normalize() forward-fills gaps, which would put a
 * flat segment covering a fifth of the path into the corpus. Every holiday
 * week would then correlate strongly with every other holiday week for a
 * reason that has nothing to do with price. Dropping them costs roughly nine
 * weeks a year and buys a corpus where a match means something.
 *
 * Days are placed by weekday, not by arrival order, so a gap can never shift
 * Friday's bars into Thursday's slot.
 */
export function buildWeeks(days: DayPath[], mode: WeekMode): WeekBuild {
  const per = DAY_BARS[mode];
  const offset = mode === "rth" ? RTH_OPEN_BAR : 0;

  const byWeek = new Map<string, (Float32Array | null)[]>();
  for (const d of days) {
    const monday = mondayOf(d.date);
    if (monday === null) continue;
    let slots = byWeek.get(monday);
    if (!slots) {
      slots = [null, null, null, null, null];
      byWeek.set(monday, slots);
    }
    slots[weekdayIndex(d.date)] = d.closes.subarray(offset, offset + per);
  }

  const weeks: DayPath[] = [];
  let incomplete = 0;

  for (const [monday, slots] of byWeek) {
    if (slots.some((s) => s === null)) {
      incomplete++;
      continue;
    }
    const closes = new Float32Array(per * 5);
    for (let i = 0; i < 5; i++) closes.set(slots[i]!, i * per);
    // Dated by its Monday, so the engine's `cand.date >= target.date`
    // lookahead guard keeps working unchanged on weeks.
    weeks.push({ date: monday, closes });
  }

  weeks.sort((a, b) => a.date.localeCompare(b.date));
  return { weeks, incomplete };
}

/** "Wed 09:30" for a bar index within a week. */
export function weekBarLabel(bar: number, mode: WeekMode): string {
  const per = DAY_BARS[mode];
  const day = Math.max(0, Math.min(4, Math.floor(bar / per)));
  const within = bar - day * per;
  const minuteOfSession = mode === "rth" ? RTH_OPEN_BAR + within : within;
  const clock = (18 * 60 + minuteOfSession) % (24 * 60);
  const hh = String(Math.floor(clock / 60)).padStart(2, "0");
  const mm = String(clock % 60).padStart(2, "0");
  return `${WEEKDAYS[day]} ${hh}:${mm}`;
}

/**
 * Axis marks: one per weekday boundary, plus the cash open inside each day
 * when the overnights are present to distinguish it from.
 */
export function weekMarks(mode: WeekMode): Mark[] {
  const per = DAY_BARS[mode];
  const out: Mark[] = [];
  for (let i = 0; i < 5; i++) {
    out.push({ bar: i * per, label: WEEKDAYS[i], strong: true });
    if (mode === "24h") {
      out.push({ bar: i * per + RTH_OPEN_BAR, label: "9:30" });
    }
  }
  return out;
}

/** Duration of a bar span, phrased in sessions when it is long enough. */
export function fmtWeekSpan(bars: number, mode: WeekMode): string {
  const per = DAY_BARS[mode];
  if (bars >= per) {
    const d = bars / per;
    return `${d.toFixed(d % 1 === 0 ? 0 : 1)}d`;
  }
  const mins = bars;
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const r = mins % 60;
  return r ? `${h}h ${r}m` : `${h}h`;
}
