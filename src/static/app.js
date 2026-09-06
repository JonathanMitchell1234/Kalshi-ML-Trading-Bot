/* BitBot shared frontend helpers. No dependencies. */
"use strict";
const App = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt$ = (x, dp = 2) => x == null || !isFinite(x) ? "—"
    : "$" + Number(x).toLocaleString(undefined, { maximumFractionDigits: dp });
  const fmtC = (x) => x == null || !isFinite(x) ? "—" : (x / 100).toLocaleString(undefined,
    { style: "currency", currency: "USD" });
  const pct = (x, dp = 1) => x == null || !isFinite(x) ? "—" : (x * 100).toFixed(dp) + "%";

  async function api(path, opts = {}) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), opts.timeoutMs || 90000);
    try {
      const r = await fetch(path, { ...opts, signal: ctl.signal });
      if (!r.ok) {
        let msg = r.statusText;
        try { msg = await r.text(); } catch { /* ignore */ }
        throw new Error(msg.slice(0, 220) || `HTTP ${r.status}`);
      }
      return await r.json();
    } finally { clearTimeout(t); }
  }
  const jget = (u) => api(u);

  function toast(msg, kind = "") {
    const box = $("toasts") || (() => {
      const d = document.createElement("div"); d.id = "toasts"; document.body.appendChild(d); return d;
    })();
    const el = document.createElement("div");
    el.className = "toast " + kind; el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 400); }, 5200);
  }

  function confirmDlg({ title, body, okLabel = "Confirm", danger = false }) {
    return new Promise((resolve) => {
      let dlg = $("app-confirm");
      if (!dlg) {
        dlg = document.createElement("dialog"); dlg.id = "app-confirm";
        dlg.innerHTML = `<h3></h3><p></p><div class="row">
          <button class="ghost" data-x="cancel">Cancel</button>
          <button data-x="ok"></button></div>`;
        document.body.appendChild(dlg);
      }
      dlg.querySelector("h3").textContent = title;
      dlg.querySelector("p").textContent = body;
      const ok = dlg.querySelector('[data-x="ok"]');
      ok.textContent = okLabel;
      ok.className = danger ? "danger-ghost" : "";
      const done = (v) => { dlg.close(); dlg.onclose = null; resolve(v); };
      dlg.querySelector('[data-x="cancel"]').onclick = () => done(false);
      ok.onclick = () => done(true);
      dlg.onclose = () => resolve(false);
      dlg.showModal();
    });
  }

  // Polling manager: skips while tab hidden, supports global pause + countdown.
  const Poller = {
    paused: false, jobs: [],
    add(fn, ms) { this.jobs.push({ fn, ms, next: 0 }); },
    tick() {
      if (this.paused || document.hidden) return;
      const now = Date.now();
      for (const j of this.jobs) {
        if (now >= j.next) {
          j.next = now + j.ms;
          Promise.resolve().then(j.fn).catch((e) => console.warn("poll", e));
        }
      }
    },
    start(sec = 5) { setInterval(() => this.tick(), sec * 1000); this.tick(); },
  };

  function sparkline(svg, points, { stroke = "#58a6ff", fill = true } = {}) {
    if (!svg || !points || points.length < 2) {
      if (svg) svg.innerHTML = `<text x="8" y="34" fill="#6e7681" font-size="12">not enough data yet</text>`;
      return;
    }
    const W = 600, H = 64, P = 6;
    const vs = points.map((p) => p.y);
    const mn = Math.min(...vs), mx = Math.max(...vs), sp = (mx - mn) || 1;
    const X = (i) => P + (i / (points.length - 1)) * (W - 2 * P);
    const Y = (v) => H - P - ((v - mn) / sp) * (H - 2 * P);
    const line = points.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
    const zero = Y(Math.min(Math.max(0, mn), mx));
    const neg = vs[vs.length - 1] < 0;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = `<line x1="${P}" y1="${zero}" x2="${W - P}" y2="${zero}" stroke="#30363d"/>` +
      (fill ? `<path d="${line} L${X(points.length - 1)},${H - P} L${P},${H - P} Z" fill="${neg ? "#f8514918" : "#3fb95018"}" stroke="none"/>` : "") +
      `<path d="${line}" fill="none" stroke="${neg ? "#f85149" : stroke}" stroke-width="2"/>` +
      `<circle cx="${X(points.length - 1)}" cy="${Y(vs[vs.length - 1])}" r="3.5" fill="${neg ? "#f85149" : stroke}"/>`;
  }

  // Conviction meter: 0..1 probability with no-trade middle band.
  function meter(el, p, conf) {
    if (!el) return;
    const lo = 0.5 - conf, hi = 0.5 + conf;
    const pctX = (v) => (Math.min(Math.max(v, 0), 1) * 100).toFixed(1) + "%";
    const zone = p == null ? "" : (p >= hi ? "buy" : (p <= lo ? "sell" : "hold"));
    const zoneTxt = p == null ? "model unavailable"
      : zone === "buy" ? `LONG zone (≥ ${(hi * 100).toFixed(0)}%)`
      : zone === "sell" ? `SHORT zone (≤ ${(lo * 100).toFixed(0)}%)` : "chop — no trade";
    el.innerHTML = `<div class="meter" role="img" aria-label="P(up) ${p == null ? "unknown" : (p * 100).toFixed(1) + "%"}, ${zoneTxt}">
        <div class="zone-sell" style="width:${pctX(lo)}"></div>
        <div class="zone-buy" style="left:${pctX(hi)};right:0"></div>
        <div class="mid" style="left:50%"></div>
        ${p == null ? "" : `<div class="needle" style="left:${pctX(p)}"></div>`}
      </div>
      <div class="meter-labels"><span>0</span><span><b>${p == null ? "—" : (p * 100).toFixed(1) + "%"} · ${zoneTxt}</b></span><span>100</span></div>`;
    return zone;
  }

  function etClock(el) {
    if (!el) return;
    try {
      el.textContent = new Intl.DateTimeFormat("en-US", {
        timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", second: "2-digit",
        hour12: false, weekday: "short",
      }).format(new Date()) + " ET";
    } catch { /* ignore */ }
  }

  function ago(iso) {
    if (!iso) return "—";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `${Math.floor(s)}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  }

  return { $, esc, fmt$, fmtC, pct, api, jget, toast, confirmDlg, Poller, sparkline, meter, etClock, ago };
})();
