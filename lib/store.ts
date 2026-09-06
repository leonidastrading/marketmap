import { DayPath } from "./engine";

/**
 * Keep the loaded corpus across reloads.
 *
 * Re-parsing a 34 MB CSV on every refresh is the actual cost, not storage --
 * so what goes in is the parsed result, as one packed Float32Array rather than
 * 1,293 small ones. Five years of sessions is about 7 MB that way, roughly a
 * fifth of the CSV, and it comes back without touching the parser.
 *
 * Everything here fails soft. Private browsing, a denied quota, or a browser
 * without IndexedDB all just mean no cache, never a broken page.
 */

const DB_NAME = "marketmap";
const DB_VERSION = 1;
const STORE = "corpus";
const KEY = "days";

export interface SavedCorpus {
  days: DayPath[];
  source: string;
  savedAt: number;
}

interface Record {
  source: string;
  savedAt: number;
  bars: number;
  dates: string[];
  closes: ArrayBuffer;
}

function open(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") return resolve(null);
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      return resolve(null);
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}

function tx<T>(
  db: IDBDatabase,
  mode: IDBTransactionMode,
  run: (s: IDBObjectStore) => IDBRequest
): Promise<T | null> {
  return new Promise((resolve) => {
    let req: IDBRequest;
    try {
      req = run(db.transaction(STORE, mode).objectStore(STORE));
    } catch {
      return resolve(null);
    }
    req.onsuccess = () => resolve(req.result as T);
    req.onerror = () => resolve(null);
  });
}

/** Pack sessions into one buffer. Requires a uniform bar count, which
 *  parseBars guarantees -- a ragged corpus is a bug, so bail rather than
 *  silently cache something the loader would misread. */
export async function saveCorpus(
  days: DayPath[],
  source: string
): Promise<boolean> {
  if (days.length === 0) return false;
  const bars = days[0].closes.length;
  if (days.some((d) => d.closes.length !== bars)) return false;

  const flat = new Float32Array(days.length * bars);
  for (let i = 0; i < days.length; i++) flat.set(days[i].closes, i * bars);

  const db = await open();
  if (!db) return false;
  const rec: Record = {
    source,
    savedAt: Date.now(),
    bars,
    dates: days.map((d) => d.date),
    closes: flat.buffer,
  };
  const ok = await tx<IDBValidKey>(db, "readwrite", (s) => s.put(rec, KEY));
  db.close();
  return ok !== null;
}

export async function loadCorpus(): Promise<SavedCorpus | null> {
  const db = await open();
  if (!db) return null;
  const rec = await tx<Record>(db, "readonly", (s) => s.get(KEY));
  db.close();
  if (!rec || !rec.closes || !rec.dates?.length) return null;

  const flat = new Float32Array(rec.closes);
  const { bars, dates } = rec;
  if (flat.length !== dates.length * bars) return null; // truncated write

  const days: DayPath[] = dates.map((date, i) => ({
    date,
    closes: flat.slice(i * bars, (i + 1) * bars),
  }));
  return { days, source: rec.source, savedAt: rec.savedAt };
}

export async function clearCorpus(): Promise<void> {
  const db = await open();
  if (!db) return;
  await tx(db, "readwrite", (s) => s.delete(KEY));
  db.close();
}
