import { normalize, scan, project, backtest, SESSION_BARS } from "../lib/engine";
import { syntheticCorpus, sessionSlot, parseBars } from "../lib/data";

// --- session slot mapping -------------------------------------------------
const t1 = Date.parse("2024-03-11T22:05:00Z"); // 18:05 ET (EDT)
const t2 = Date.parse("2024-03-12T13:30:00Z"); // 09:30 ET (EDT)
const t3 = Date.parse("2024-01-08T23:00:00Z"); // 18:00 ET (EST)
console.log("slot 18:05 EDT ->", sessionSlot(t1));
console.log("slot 09:30 EDT ->", sessionSlot(t2), "(expect bar 930)");
console.log("slot 18:00 EST ->", sessionSlot(t3), "(expect bar 0)");
console.log(
  "maintenance 17:30 ET ->",
  sessionSlot(Date.parse("2024-03-12T21:30:00Z")),
  "(expect null)"
);

// --- csv round trip -------------------------------------------------------
const csv = [
  "ts_event,open,high,low,close,volume",
  "2024-03-11T22:00:00Z,5100,5101,5099,5100.25,120",
  "2024-03-11T22:01:00Z,5100,5102,5100,5101.50,90",
  "2024-03-12T13:30:00Z,5110,5115,5108,5112.00,4000",
].join("\n");
const parsed = parseBars(csv);
console.log(
  "\ncsv -> days:",
  parsed.map((d) => d.date),
  "bar0:",
  parsed[0].closes[0],
  "bar930:",
  parsed[0].closes[930]
);

// --- engine on structureless data ----------------------------------------
console.log("\nbuilding 1250 synthetic sessions...");
const raw = syntheticCorpus(1250);
const corpus = raw.map(normalize);
console.log("corpus:", corpus.length, "days x", SESSION_BARS, "bars");

const target = corpus[corpus.length - 1];
const history = corpus.slice(0, -1);
const t = 1020; // ~12:00 ET, 90 minutes into the RTH afternoon

for (const W of [30, 60, 120, 240]) {
  const res = scan(target, history, {
    t,
    windowBars: W,
    anchored: false,
    topK: 25,
    metric: "pearson",
  });
  console.log(
    `window=${String(W).padStart(3)}  best +corr=${res.positive[0].corr.toFixed(4)}` +
      `  best -corr=${res.negative[0].corr.toFixed(4)}` +
      `  25th best +=${res.positive[24].corr.toFixed(4)}`
  );
}

const res = scan(target, history, {
  t,
  windowBars: 120,
  anchored: false,
  topK: 25,
  metric: "pearson",
});
const proj = project(target, history, res.positive, t);
console.log(
  "\nprojection horizon:",
  proj.mean.length,
  " anchor:",
  target.closes[t].toFixed(2),
  " ensemble terminal:",
  proj.mean[proj.mean.length - 1].toFixed(2),
  " p10/p90:",
  proj.p10[proj.p10.length - 1].toFixed(2),
  "/",
  proj.p90[proj.p90.length - 1].toFixed(2)
);
console.log("actual terminal:", target.closes[SESSION_BARS - 1].toFixed(2));

// --- vol scaling sanity ---------------------------------------------------
const unscaled = project(target, history, res.positive, t, false);
const spanScaled =
  proj.p90[proj.p90.length - 1] - proj.p10[proj.p10.length - 1];
const spanRaw =
  unscaled.p90[unscaled.p90.length - 1] - unscaled.p10[unscaled.p10.length - 1];
console.log(
  `vol scaling: fan width ${spanRaw.toFixed(2)} -> ${spanScaled.toFixed(2)}`
);

// --- walk-forward on pure noise -------------------------------------------
console.log("\nwalk-forward backtest on driftless random walks:");
const bt = backtest(corpus, {
  t,
  windowBars: 120,
  anchored: false,
  topK: 25,
  metric: "pearson",
  startIndex: 900,
  maxDays: 300,
});
console.log(`  days tested       ${bt.n}`);
console.log(`  hit rate          ${(bt.hitRate * 100).toFixed(1)}%`);
console.log(`  always-long       ${(bt.baselineLong * 100).toFixed(1)}%`);
console.log(`  momentum baseline ${(bt.baselineMomentum * 100).toFixed(1)}%`);
console.log(`  mean |best corr|  ${bt.meanBestCorr.toFixed(4)}`);
console.log(`  mean path corr    ${bt.meanPathCorr.toFixed(4)}`);
