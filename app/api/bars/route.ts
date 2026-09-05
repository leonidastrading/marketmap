import { NextRequest } from "next/server";

/**
 * Proxy for Databento historical OHLCV-1m, so the API key never reaches the
 * browser.
 *
 * GET /api/bars?start=2026-09-03&end=2026-09-05&symbol=ES.c.0
 *
 * Note on freshness: Databento's *historical* API serves data older than about
 * 24 hours. Polling this route will give you completed sessions, which is what
 * you want for replay and for building the corpus, but it will not give you the
 * bar that just printed. For a live tail you need a persistent process holding
 * a Databento Live session — see README, "Live tail".
 */

export const runtime = "nodejs";

const HIST = "https://hist.databento.com/v0/timeseries.get_range";

export async function GET(req: NextRequest) {
  const key = process.env.DATABENTO_API_KEY;
  if (!key) {
    return Response.json(
      {
        error: "DATABENTO_API_KEY is not set.",
        fix: "Add it in Vercel → Project → Settings → Environment Variables, then redeploy.",
      },
      { status: 501 }
    );
  }

  const p = req.nextUrl.searchParams;
  const start = p.get("start");
  const end = p.get("end");
  const symbol = p.get("symbol") ?? "ES.c.0";

  if (!start) {
    return Response.json({ error: "start is required (YYYY-MM-DD)." }, { status: 400 });
  }

  const body = new URLSearchParams({
    dataset: "GLBX.MDP3",
    symbols: symbol,
    schema: "ohlcv-1m",
    stype_in: "continuous",
    encoding: "csv",
    start,
  });
  if (end) body.set("end", end);

  const auth = Buffer.from(`${key}:`).toString("base64");

  try {
    const r = await fetch(HIST, {
      method: "POST",
      headers: {
        Authorization: `Basic ${auth}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body,
    });

    if (!r.ok) {
      const text = await r.text();
      return Response.json(
        { error: `Databento returned ${r.status}`, detail: text.slice(0, 400) },
        { status: r.status }
      );
    }

    return new Response(r.body, {
      headers: {
        "Content-Type": "text/csv",
        "Cache-Control": "public, max-age=300",
      },
    });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "Upstream request failed." },
      { status: 502 }
    );
  }
}
