/* Interactive results page: plain SVG charts, no dependencies. Data: data/results.json
   (written by export_site_data.py). */
"use strict";

const NS = "http://www.w3.org/2000/svg";
const POLICIES = ["LRU", "LFU", "FIFO", "MRU"];
const PCOL = ["var(--p-lru)", "var(--p-lfu)", "var(--p-fifo)", "var(--p-mru)"];

// colour follows the method, never its rank
const METHOD = {
  "DQN": { label: "Double DQN", color: "var(--m-dqn)", width: 2.5 },
  "Q-Learning": { label: "Q-learning", color: "var(--m-q)" },
  "ARC": { label: "ARC", color: "var(--m-arc)" },
  "FTL": { label: "FTL", color: "var(--m-ftl)" },
  "Fixed-LRU": { label: "Fixed LRU", color: "var(--m-lru)" },
  "Belady": { label: "Belady (offline)", color: "var(--m-opt)", dash: "5 4" },
};
const LINE_METHODS = Object.keys(METHOD);
const label = (m) => (METHOD[m] ? METHOD[m].label : m.replace("Fixed-", "Fixed "));
const colorOf = (m) => (METHOD[m] ? METHOD[m].color : "var(--m-other)");

const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const $ = (id) => document.getElementById(id);
let D = null;

/* ---------------------------------------------------------------- helpers */
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k === "style") e.setAttribute("style", v);
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, attrs = {}, html) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (html !== undefined) e.innerHTML = html;
  return e;
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function newSvg(host, height) {
  host.innerHTML = "";
  const w = Math.max(320, host.clientWidth);
  const svg = el("svg", { viewBox: `0 0 ${w} ${height}`, width: w, height }, host);
  return { svg, w };
}

const tip = {
  node: null,
  show(ev, html) {
    this.node = this.node || $("tip");
    this.node.innerHTML = html;
    this.node.style.opacity = 1;
    const pad = 14, r = this.node.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    this.node.style.left = x + "px";
    this.node.style.top = y + "px";
  },
  hide() { if (this.node) this.node.style.opacity = 0; },
};
const keyHtml = (c, dash) => `<i class="key${dash ? " dash" : ""}" style="border-top-color:${c}"></i>`;
const swHtml = (c) => `<i class="key sw" style="background:${c}"></i>`;
const tipRow = (k, name, val) => `<div class="r"><span class="n">${k}${name}</span><span>${val}</span></div>`;

/* segmented control; returns getter */
function seg(host, options, initial, onChange) {
  host.innerHTML = "";
  let cur = initial;
  const btns = options.map(([value, text]) => {
    const b = h("button", { type: "button", "aria-pressed": String(value === cur) }, text);
    b.onclick = () => {
      if (b.disabled) return;
      cur = value;
      btns.forEach((x, i) => x.setAttribute("aria-pressed", String(options[i][0] === cur)));
      onChange(cur);
    };
    host.appendChild(b);
    return b;
  });
  return {
    get: () => cur,
    disable(values) { btns.forEach((b, i) => (b.disabled = values.includes(options[i][0]))); },
    set(v) { cur = v; btns.forEach((x, i) => x.setAttribute("aria-pressed", String(options[i][0] === cur))); },
  };
}
/* toggle chips for line series */
function chips(host, methods, on, onChange) {
  host.innerHTML = "";
  const state = new Set(on);
  methods.forEach((m) => {
    const b = h("button", { type: "button", class: "chip", "aria-pressed": String(state.has(m)) },
      keyHtml(colorOf(m), METHOD[m] && METHOD[m].dash) + label(m));
    b.onclick = () => {
      state.has(m) ? state.delete(m) : state.add(m);
      b.setAttribute("aria-pressed", String(state.has(m)));
      onChange();
    };
    host.appendChild(b);
  });
  return () => methods.filter((m) => state.has(m));
}
function table(host, head, rows, numCols = []) {
  const t = h("table");
  t.appendChild(h("thead", {}, "<tr>" + head.map((c, i) =>
    `<th class="${numCols.includes(i) ? "num" : ""}">${c}</th>`).join("") + "</tr>"));
  const tb = h("tbody");
  rows.forEach((r) => {
    const tr = h("tr", r.cls ? { class: r.cls } : {});
    tr.innerHTML = r.cells.map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${c}</td>`).join("");
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  host.innerHTML = "";
  host.appendChild(t);
}

/* ------------------------------------------------------------ line chart */
/* series: [{key, color, dash, width, xs, ys}] sharing an x grid; opts: x/y formatters, bands */
function lineChart(host, series, o) {
  const H = o.height || 280;
  const { svg, w } = newSvg(host, H);
  const m = { l: 46, r: 14, t: o.bands ? 22 : 8, b: o.xLabel ? 38 : 24 };
  const xs = o.xs;
  const x0 = o.xMin ?? xs[0], x1 = o.xMax ?? xs[xs.length - 1];
  let ylo = o.yMin, yhi = o.yMax;
  if (ylo === undefined || yhi === undefined) {
    const all = series.flatMap((s) => s.ys).filter((v) => v !== null && !isNaN(v));
    const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 0.05;
    ylo = ylo ?? lo - pad; yhi = yhi ?? hi + pad;
  }
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (w - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - ylo) / (yhi - ylo)) * (H - m.t - m.b);

  (o.bands || []).forEach((b, i) => {
    if (i % 2 === 0) el("rect", { x: X(b.x0), y: m.t, width: X(b.x1) - X(b.x0), height: H - m.t - m.b, style: "fill:var(--band)" }, svg);
    el("text", { x: (X(b.x0) + X(b.x1)) / 2, y: m.t - 7, "text-anchor": "middle", class: "band-lbl", text: b.label }, svg);
  });
  niceTicks(ylo, yhi, 5).forEach((t) => {
    el("line", { x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
    el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end", text: o.yFmt ? o.yFmt(t) : t }, svg);
  });
  el("line", { x1: m.l, x2: w - m.r, y1: H - m.b, y2: H - m.b, class: "baseline" }, svg);
  const xt = o.xTicks || niceTicks(x0, x1, Math.max(3, Math.floor((w - m.l) / 90)));
  xt.forEach((t) => el("text", { x: X(t), y: H - m.b + 15, "text-anchor": "middle", text: o.xFmt ? o.xFmt(t) : t }, svg));
  if (o.xLabel) el("text", { x: (m.l + w - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "lbl", text: o.xLabel }, svg);

  series.forEach((s) => {
    let d = "";
    s.ys.forEach((v, i) => { if (v === null || isNaN(v)) return; d += (d ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(v).toFixed(1); });
    el("path", { d, style: `fill:none;stroke:${s.color};stroke-width:${s.width || 2};stroke-linejoin:round;stroke-linecap:round;${s.dash ? "stroke-dasharray:" + s.dash : ""}` }, svg);
    if (o.markers) s.ys.forEach((v, i) => el("circle", { cx: X(xs[i]), cy: Y(v), r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2` }, svg));
  });

  // hover layer: crosshair + tooltip with every visible series at that x
  const g = el("g", { style: "pointer-events:none" }, svg);
  const hair = el("line", { y1: m.t, y2: H - m.b, class: "xhair", style: "opacity:0" }, g);
  const dots = series.map((s) => el("circle", { r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2;opacity:0` }, g));
  const hit = el("rect", { x: m.l, y: 0, width: w - m.l - m.r, height: H - m.b, style: "fill:transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * w;
    const xv = x0 + ((px - m.l) / (w - m.l - m.r)) * (x1 - x0);
    let i = 0, best = Infinity;
    xs.forEach((x, j) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; i = j; } });
    hair.setAttribute("x1", X(xs[i])); hair.setAttribute("x2", X(xs[i])); hair.style.opacity = 1;
    series.forEach((s, k) => {
      const v = s.ys[i];
      if (v === null || isNaN(v)) { dots[k].style.opacity = 0; return; }
      dots[k].setAttribute("cx", X(xs[i])); dots[k].setAttribute("cy", Y(v)); dots[k].style.opacity = 1;
    });
    const rows = series.map((s) => [s, s.ys[i]]).filter(([, v]) => v !== null && !isNaN(v)).sort((a, b) => b[1] - a[1])
      .map(([s, v]) => tipRow(keyHtml(s.color, s.dash), s.label, o.yFmt ? o.yFmt(v) : v.toFixed(3))).join("");
    tip.show(ev, `<div class="t">${o.tipTitle ? o.tipTitle(xs[i], i) : xs[i]}</div>${rows}`);
  });
  hit.addEventListener("pointerleave", () => { hair.style.opacity = 0; dots.forEach((d) => (d.style.opacity = 0)); tip.hide(); });
  return { X, m, w };
}

/* ---------------------------------------------------------------- tiles */
function renderTiles() {
  const s3 = byMethod(D.summary.a3), s4 = byMethod(D.summary.a4);
  const gap = (s4.DQN.hit_rate - s3["Fixed-LRU"].hit_rate) / (s3.Belady.hit_rate - s3["Fixed-LRU"].hit_rate);
  const tiles = [
    ["Double DQN hit rate (+MRU)", pct(s4.DQN.hit_rate), `<b>+${(100 * (s4.DQN.hit_rate - s4.ARC.hit_rate)).toFixed(1)} pp</b> vs ARC (${pct(s4.ARC.hit_rate)})`],
    ["Double DQN hit rate (3 actions)", pct(s3.DQN.hit_rate), `<b>+${(100 * (s3.DQN.hit_rate - s3.ARC.hit_rate)).toFixed(2)} pp</b> vs ARC, better on 18/20 traces`],
    ["AMAT reduction vs Fixed LRU", (100 * (1 - s4.DQN.amat / s4["Fixed-LRU"].amat)).toFixed(1) + "%", `${s4["Fixed-LRU"].amat.toFixed(1)} → ${s4.DQN.amat.toFixed(1)} time units (+MRU)`],
    ["Gap to Belady's optimum closed", pct(gap, 0), `LRU ${pct(s3["Fixed-LRU"].hit_rate)} → DQN ${pct(s4.DQN.hit_rate)} of ${pct(s3.Belady.hit_rate)}`],
  ];
  $("tiles").innerHTML = tiles.map(([l, v, d]) =>
    `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="delta">${d}</div></div>`).join("");
}
const byMethod = (rows) => Object.fromEntries(rows.map((r) => [r.method, r]));

/* ---------------------------------------------------------------- overall */
const OV = { act: "a3", metric: "hit_rate", size: 100 };
const METRICS = {
  hit_rate: { name: "Hit rate", fmt: (v) => pct(v, 2), higher: true },
  amat: { name: "AMAT", fmt: (v) => v.toFixed(2), higher: false },
  post_change_hit_rate: { name: "Post-change hit rate", fmt: (v) => pct(v, 2), higher: true },
};
function renderOverall() {
  const rows = (OV.act === "a4" ? D.summary.a4 : (OV.size === 100 ? D.summary.a3 : D.summary.a3_sizes.filter((r) => r.cache_size === OV.size)));
  const M = METRICS[OV.metric];
  const data = rows.map((r) => ({ ...r, v: r[OV.metric] })).sort((a, b) => (M.higher ? b.v - a.v : a.v - b.v));
  const host = $("ov-chart");
  const rowH = 26, H = data.length * rowH + 30;
  const { svg, w } = newSvg(host, H);
  const m = { l: 128, r: 70, t: 4, b: 22 };
  const vmax = Math.max(...data.map((d) => d.v + (OV.metric === "hit_rate" ? d.hit_rate_std : 0)));
  const X = (v) => m.l + (v / vmax) * (w - m.l - m.r);
  niceTicks(0, vmax, 5).forEach((t) => {
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "gridline" }, svg);
    el("text", { x: X(t), y: H - 6, "text-anchor": "middle", text: OV.metric === "amat" ? t : pct(t, 0) }, svg);
  });
  el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "baseline" }, svg);
  data.forEach((d, i) => {
    const y = m.t + i * rowH + (rowH - 16) / 2, bw = Math.max(1, X(d.v) - m.l);
    const g = el("g", {}, svg);
    el("text", { x: m.l - 8, y: y + 12, "text-anchor": "end", class: "lbl", text: label(d.method) }, g);
    // bar: 16px thick, 4px rounded data-end, square at the baseline
    const r = Math.min(4, bw / 2);
    el("path", { d: `M${m.l},${y}H${m.l + bw - r}Q${m.l + bw},${y} ${m.l + bw},${y + r}V${y + 16 - r}Q${m.l + bw},${y + 16} ${m.l + bw - r},${y + 16}H${m.l}Z`, style: `fill:${colorOf(d.method)}${d.method === "Belady" ? ";opacity:0.45" : ""}` }, g);
    let end = m.l + bw;
    if (OV.metric === "hit_rate") {
      const a = X(Math.max(0, d.v - d.hit_rate_std)), b = X(d.v + d.hit_rate_std);
      el("line", { x1: a, x2: b, y1: y + 8, y2: y + 8, style: "stroke:var(--ink2);stroke-width:1" }, g);
      end = b;
    }
    el("text", { x: end + 6, y: y + 12, class: "val", text: M.fmt(d.v) }, g);
    const hit = el("rect", { x: 0, y: m.t + i * rowH, width: w, height: rowH, style: "fill:transparent" }, g);
    hit.addEventListener("pointermove", (ev) => tip.show(ev, `<div class="t">${label(d.method)}</div>` +
      tipRow("", "Hit rate", pct(d.hit_rate, 2) + " ± " + pct(d.hit_rate_std, 1)) + tipRow("", "AMAT", d.amat.toFixed(2)) +
      tipRow("", "Post-change", pct(d.post_change_hit_rate, 2))));
    hit.addEventListener("pointerleave", () => tip.hide());
  });
  $("ov-legend").innerHTML = ["DQN", "Q-Learning", "ARC", "FTL", "Fixed-LRU", "Belady"].map((k) => `<span><i class="key sw" style="background:${colorOf(k)}${k === "Belady" ? ";opacity:0.45" : ""}"></i>${label(k)}</span>`).join("") +
    `<span>${swHtml("var(--m-other)")}other baselines</span>` + (OV.metric === "hit_rate" ? `<span>${keyHtml("var(--ink2)")}± std over traces</span>` : "");
  const s = byMethod(rows);
  const best = data.find((d) => d.method !== "Belady");
  $("ov-note").innerHTML = `Best online method: <b>${label(best.method)}</b> (${M.fmt(best.v)}). ` +
    (OV.act === "a4" ? "Random, FTL and Phase-Oracle here select among 4 actions. " : "") +
    (OV.size !== 100 && OV.act === "a3" ? `Agents were trained at C=100; at C=${OV.size} DQN ${s.DQN.hit_rate >= s.ARC.hit_rate ? "stays ahead of" : "falls just behind"} ARC.` : "");
  table($("ov-table"), ["Method", "Hit rate", "± std", "AMAT", "Post-change"],
    data.map((d) => ({ cells: [label(d.method), pct(d.hit_rate, 2), d.hit_rate_std.toFixed(3), d.amat.toFixed(2), pct(d.post_change_hit_rate, 2)] })), [1, 2, 3, 4]);
}

/* ---------------------------------------------------------------- timeline */
const TL = { scn: "drift_loop_stack", act: "a4", agent: "DQN" };
let tlOn = () => [];
function renderTimeline() {
  const sc = D.scenarios[TL.act][TL.scn];
  if (!sc) return;
  const W = sc.window, n = sc.methods.DQN.hit_rate.length;
  const xs = Array.from({ length: n }, (_, i) => (i + 0.5) * W);
  const series = tlOn().filter((k) => sc.methods[k]).map((k) => ({
    key: k, label: label(k), color: colorOf(k), dash: METHOD[k].dash, width: METHOD[k].width, ys: sc.methods[k].hit_rate,
  }));
  const bands = sc.phases.map(([s, e, r]) => ({ x0: s, x1: e, label: r }));
  const acts = sc.methods[TL.agent].actions;
  const { X, m, w } = lineChart($("tl-chart"), series, {
    xs, xMin: 0, xMax: n * W, yMin: 0, yMax: 1, yFmt: (v) => pct(v, 0), bands, height: 300,
    xFmt: (v) => (v >= 1000 ? v / 1000 + "k" : v),
    tipTitle: (x, i) => `Window ${i} · requests ${i * W}–${(i + 1) * W - 1}` +
      `<div style="font-weight:400;color:var(--ink2)">${bands.find((b) => x >= b.x0 && x < b.x1).label} phase · ${label(TL.agent)} uses ${POLICIES[acts[i]]}</div>`,
  });
  // action strip on the same x scale
  const host = $("tl-strip");
  const { svg } = newSvg(host, 46);
  el("text", { x: m.l - 6, y: 18, "text-anchor": "end", class: "lbl", text: label(TL.agent).replace("Double ", "") }, svg);
  acts.forEach((a, i) => {
    const r = el("rect", { x: X(i * W), y: 6, width: Math.max(0.5, X((i + 1) * W) - X(i * W)), height: 18, style: `fill:${PCOL[a]}` }, svg);
    r.addEventListener("pointermove", (ev) => tip.show(ev, `<div class="t">Window ${i}</div>${tipRow(swHtml(PCOL[a]), "Policy", POLICIES[a])}`));
    r.addEventListener("pointerleave", () => tip.hide());
  });
  niceTicks(0, n * W, Math.max(3, Math.floor((w - m.l) / 90))).forEach((t) =>
    el("text", { x: X(t), y: 40, "text-anchor": "middle", text: t >= 1000 ? t / 1000 + "k" : t }, svg));
  const nA = TL.act === "a4" ? 4 : 3;
  $("tl-plegend").innerHTML = `<span style="color:var(--ink2)">Strip colours (policy chosen):</span>` +
    POLICIES.slice(0, nA).map((p, i) => `<span>${swHtml(PCOL[i])}${p}</span>`).join("") + `<span style="margin-left:auto">x axis: request index</span>`;
  // per-phase summary for the note
  const mean = (arr, s, e) => arr.slice(s, e).reduce((a, b) => a + b, 0) / (e - s);
  const parts = sc.phases.map(([s, e, r]) => {
    const i0 = Math.floor(s / W), i1 = Math.floor(e / W);
    const dq = mean(sc.methods.DQN.hit_rate, i0, i1), arc = mean(sc.methods.ARC.hit_rate, i0, i1);
    return `${r}: DQN ${pct(dq, 0)} vs ARC ${pct(arc, 0)}`;
  });
  $("tl-note").textContent = "Phase means: " + parts.join(" · ") + ".";
}

/* ---------------------------------------------------------------- per-regime heatmap */
const RG = { act: "a4" };
function renderRegimes() {
  const rows = D.per_regime[RG.act];
  const regs = D.config.regimes;
  const order = ["DQN", "Q-Learning", "ARC", "FTL", "Phase-Oracle", "Fixed-LRU", "Fixed-LFU", "Fixed-FIFO", "Fixed-MRU", "Random", "Belady"];
  const methods = order.filter((mm) => rows.some((r) => r.method === mm));
  const val = (mm, rg) => (rows.find((r) => r.method === mm && r.regime === rg) || {}).hit_rate;
  const best = Object.fromEntries(regs.map((rg) => {
    let b = null;
    methods.filter((mm) => mm !== "Belady").forEach((mm) => { if (b === null || val(mm, rg) > val(b, rg)) b = mm; });
    return [rg, b];
  }));
  const t = h("table", { class: "heat" });
  t.innerHTML = `<thead><tr><th>Method</th>${regs.map((r) => `<th>${r}</th>`).join("")}</tr></thead>`;
  const tb = h("tbody");
  methods.forEach((mm) => {
    const tr = h("tr");
    tr.appendChild(h("td", {}, (METHOD[mm] ? keyHtml(colorOf(mm), METHOD[mm].dash) + " " : "") + label(mm)));
    regs.forEach((rg) => {
      const v = val(mm, rg);
      const a = 0.06 + 0.94 * Math.min(1, Math.max(0, v / 0.85));
      const td = h("td", { class: "cell" + (best[rg] === mm ? " best" : "") }, (100 * v).toFixed(1));
      td.style.background = `rgba(var(--heat), ${a.toFixed(3)})`;
      td.style.color = a > 0.55 ? "#ffffff" : "var(--ink)";
      td.addEventListener("pointermove", (ev) => tip.show(ev, `<div class="t">${label(mm)} · ${rg}</div>${tipRow("", "Hit rate", pct(v, 2))}${tipRow("", "vs Belady", pct(val("Belady", rg), 1))}`));
      td.addEventListener("pointerleave", () => tip.hide());
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  $("rg-table").innerHTML = "";
  $("rg-table").appendChild(t);
  $("rg-note").innerHTML = RG.act === "a4"
    ? `With MRU available, the gain is concentrated in <b>loop</b> phases: DQN ${pct(val("DQN", "loop"))} vs ARC ${pct(val("ARC", "loop"))} (Belady ${pct(val("Belady", "loop"))}). The agent lets LRU/LFU refill the cache with loop blocks, then switches to MRU to pin them. Fixed MRU fails because it pins stale blocks from earlier phases.`
    : `With 3 actions, DQN's advantage over Fixed LRU comes mostly from <b>scan</b> (${pct(val("DQN", "scan"))} vs ${pct(val("Fixed-LRU", "scan"))}) and <b>zipf</b> (${pct(val("DQN", "zipf"))} vs ${pct(val("Fixed-LRU", "zipf"))}) phases, where it switches to aged LFU.`;
}

/* ---------------------------------------------------------------- action distribution */
const AC = { agent: "DQN", act: "a3" };
function renderActions() {
  const rows = D.actions[AC.act].filter((r) => r.method === AC.agent);
  const regs = D.config.regimes.filter((rg) => rows.some((r) => r.regime === rg));
  const nA = AC.act === "a4" ? 4 : 3;
  const host = $("ac-chart");
  const rowH = 34, H = regs.length * rowH + 26;
  const { svg, w } = newSvg(host, H);
  const m = { l: 60, r: 12, t: 4, b: 20 };
  const X = (v) => m.l + v * (w - m.l - m.r);
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => el("text", { x: X(t), y: H - 4, "text-anchor": "middle", text: pct(t, 0) }, svg));
  regs.forEach((rg, i) => {
    const r = rows.find((x) => x.regime === rg);
    const y = m.t + i * rowH + 5;
    el("text", { x: m.l - 8, y: y + 15, "text-anchor": "end", class: "lbl", text: rg }, svg);
    let acc = 0;
    POLICIES.slice(0, nA).forEach((p, k) => {
      const f = r[p] || 0;
      if (f <= 0) return;
      const xa = X(acc), xb = X(acc + f);
      const width = Math.max(0, xb - xa - 2); // 2px surface gap between segments
      const g = el("g", {}, svg);
      el("rect", { x: xa, y, width, height: 22, rx: 3, style: `fill:${PCOL[k]}` }, g);
      if (width > 38) el("text", { x: xa + width / 2, y: y + 15, "text-anchor": "middle", style: `fill:${k === 3 ? "#0b0b0b" : "#ffffff"};font-size:11.5px`, text: pct(f, 0) }, g);
      g.addEventListener("pointermove", (ev) => tip.show(ev, `<div class="t">${label(AC.agent)} · ${rg}</div>${tipRow(swHtml(PCOL[k]), p, pct(f, 1))}`));
      g.addEventListener("pointerleave", () => tip.hide());
      acc += f;
    });
  });
  $("ac-legend").innerHTML = POLICIES.slice(0, nA).map((p, i) => `<span>${swHtml(PCOL[i])}${p}</span>`).join("");
  const top = (rg) => { const r = rows.find((x) => x.regime === rg); return POLICIES.slice(0, nA).reduce((a, b) => (r[b] > r[a] ? b : a)); };
  $("ac-note").textContent = `Most-used policy per regime: ` + regs.map((rg) => `${rg} → ${top(rg)}`).join(", ") + ".";
}

/* ---------------------------------------------------------------- recovery & sizes */
const RC = { act: "a3" };
let rcOn = () => [], szOn = () => [];
function renderRecovery() {
  const rows = D.post_change[RC.act];
  const ks = [...new Set(rows.map((r) => r.window_after_change))].sort((a, b) => a - b);
  const on = rcOn().filter((k) => rows.some((r) => r.method === k));
  const series = on.map((k) => ({ key: k, label: label(k), color: colorOf(k), dash: METHOD[k].dash, width: METHOD[k].width,
    ys: ks.map((x) => rows.find((r) => r.method === k && r.window_after_change === x).hit_rate) }));
  lineChart($("rc-chart"), series, { xs: ks, xTicks: ks, yFmt: (v) => pct(v, 1), height: 250, markers: true,
    xLabel: "window after phase change", tipTitle: (x) => `Window ${x} after change` });
  table($("rc-table"), ["Method", ...ks.map(String)], on.map((k, i) => ({ cells: [label(k), ...series[i].ys.map((v) => pct(v, 1))] })), ks.map((_, i) => i + 1));
}
function renderSizes() {
  const rows = D.summary.a3_sizes;
  const Cs = [...new Set(rows.map((r) => r.cache_size))].sort((a, b) => a - b);
  const on = szOn();
  const series = on.map((k) => ({ key: k, label: label(k), color: colorOf(k), dash: METHOD[k].dash, width: METHOD[k].width,
    ys: Cs.map((c) => rows.find((r) => r.method === k && r.cache_size === c).hit_rate) }));
  lineChart($("sz-chart"), series, { xs: Cs, xTicks: Cs, xMin: 40, xMax: 210, yFmt: (v) => pct(v, 1), height: 250, markers: true,
    xLabel: "cache size C (blocks)", tipTitle: (x) => `C = ${x}` });
  table($("sz-table"), ["Method", ...Cs.map((c) => "C=" + c)], on.map((k, i) => ({ cells: [label(k), ...series[i].ys.map((v) => pct(v, 2))] })), Cs.map((_, i) => i + 1));
}

/* ---------------------------------------------------------------- training */
const RUNS = {
  dqn_a3: { label: "DQN · 3 actions", color: "var(--m-dqn)" },
  dqn_a4: { label: "DQN · +MRU", color: "var(--m-dqn)", dash: "5 4" },
  qlearn_a3: { label: "Q-learning · 3 actions", color: "var(--m-q)" },
  qlearn_a4: { label: "Q-learning · +MRU", color: "var(--m-q)", dash: "5 4" },
};
let trOn = new Set(Object.keys(RUNS));
function renderTraining() {
  const runs = Object.keys(RUNS).filter((k) => trOn.has(k) && D.training[k]);
  const maxEp = Math.max(...Object.values(D.training).filter(Boolean).map((t) => t.episode.length));
  const xs = Array.from({ length: maxEp }, (_, i) => i + 1);
  const smooth = (a) => a.map((_, i) => { const s = a.slice(Math.max(0, i - 14), i + 1); return s.reduce((p, q) => p + q, 0) / s.length; });
  const ret = runs.map((k) => {
    const t = D.training[k], sm = smooth(t.return);
    return { ...RUNS[k], ys: xs.map((x) => (x <= sm.length ? sm[x - 1] : null)) };
  });
  lineChart($("tr-return"), ret, { xs, yFmt: (v) => v.toFixed(0), height: 230, xLabel: "episode", tipTitle: (x) => `Episode ${x}` });
  const vEps = [...new Set(runs.flatMap((k) => D.training[k].val_episode))].sort((a, b) => a - b);
  const val = runs.map((k) => {
    const t = D.training[k];
    return { ...RUNS[k], ys: vEps.map((e) => { const j = t.val_episode.indexOf(e); return j < 0 ? null : t.val_hit_rate[j]; }) };
  });
  lineChart($("tr-val"), val, { xs: vEps, xMin: 0, xMax: maxEp, yFmt: (v) => pct(v, 0), height: 230, xLabel: "episode", markers: false, tipTitle: (x) => `Episode ${x}` });
}

/* ---------------------------------------------------------------- static tables */
function renderStats() {
  // computed by re-running evaluate.run_all on the committed checkpoints (scipy.stats.wilcoxon)
  const rows = [
    ["DQN − ARC", "3", "+0.38", "[+0.19, +0.57]", "18/20", "8.5×10⁻⁴"],
    ["DQN − FTL", "3", "+1.77", "[+1.28, +2.26]", "20/20", "1.9×10⁻⁶"],
    ["DQN − Fixed LRU", "3", "+2.79", "[+2.33, +3.25]", "20/20", "1.9×10⁻⁶"],
    ["DQN − Phase-Oracle", "3", "+0.70", "[+0.45, +0.95]", "18/20", "1.9×10⁻⁵"],
    ["Q-learning − ARC", "3", "−0.32", "[−0.60, −0.04]", "6/20", "0.04"],
    ["Q-learning − FTL", "3", "+1.06", "[+0.65, +1.48]", "18/20", "1.3×10⁻⁵"],
    ["DQN − ARC", "4", "+6.46", "[+4.52, +8.41]", "19/20", "3.8×10⁻⁶"],
    ["DQN − FTL", "4", "+7.43", "[+5.72, +9.14]", "20/20", "1.9×10⁻⁶"],
    ["Q-learning − ARC", "4", "+2.97", "[+0.99, +4.96]", "15/20", "5.6×10⁻³"],
  ];
  table($("st-table"), ["Comparison", "Actions", "Mean diff (pp)", "95% CI (pp)", "Traces won", "Wilcoxon p"],
    rows.map((r) => ({ cells: r })), [1, 2, 3, 4, 5]);
}

const FEATURE_DOC = {
  ghost_lru: ["Shadow caches", "Hit rate of a fixed-LRU shadow cache, last window"],
  ghost_lfu: ["Shadow caches", "Same, fixed aged-LFU"],
  ghost_fifo: ["Shadow caches", "Same, fixed FIFO"],
  ghost_mru: ["Shadow caches", "Same, fixed MRU"],
  ghost_lru_ema: ["Shadow caches", "EMA (β = 0.8) of the LRU shadow hit rate"],
  ghost_lfu_ema: ["Shadow caches", "EMA of the LFU shadow hit rate"],
  ghost_fifo_ema: ["Shadow caches", "EMA of the FIFO shadow hit rate"],
  ghost_mru_ema: ["Shadow caches", "EMA of the MRU shadow hit rate"],
  unique_ratio: ["Stream", "Distinct keys / W"],
  seq_ratio: ["Stream", "Fraction with key<sub>i</sub> = key<sub>i−1</sub> + 1 (scan detector)"],
  new_ratio: ["Stream", "Fraction of first-ever references"],
  reuse_short: ["Stream", "Fraction with reuse interval ≤ C"],
  reuse_mid: ["Stream", "Fraction with C < reuse interval ≤ 4C"],
  entropy: ["Stream", "Popularity entropy / ln W"],
  jaccard: ["Stream", "Key-set overlap with the previous window (phase-change cue)"],
  own_hr: ["Agent", "Agent's own hit rate, last window"],
  own_hr_ema: ["Agent", "EMA of the agent's hit rate"],
  frac_freq1: ["Agent", "Fraction of resident blocks never re-referenced"],
  prev_LRU: ["Agent", "Previous action one-hot: LRU"],
  prev_LFU: ["Agent", "Previous action one-hot: LFU"],
  prev_FIFO: ["Agent", "Previous action one-hot: FIFO"],
  prev_MRU: ["Agent", "Previous action one-hot: MRU"],
};
function renderFormulation() {
  const c = D.config;
  const feats = [...c.trace_features, ...c.agent_features];
  $("f-dim").textContent = c.obs_dim;
  table($("f-state"), ["#", "Feature", "Group", "Meaning"],
    feats.map((f, i) => ({ cells: [i, `<code>${f}</code>`, (FEATURE_DOC[f] || ["", ""])[0], (FEATURE_DOC[f] || ["", ""])[1]] })), [0]);
  $("f-actions").innerHTML = POLICIES.map((p, i) => `<span>${swHtml(PCOL[i])}${i}: ${p}${i === 3 ? " (4-action setting)" : ""}</span>`).join("");
  const kv = (id, rows) => { $(id).innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join(""); };
  const nEp = (k) => (D.training[k] ? D.training[k].episode.length : "–");
  const dq = c.dqn;
  kv("f-env", [
    ["Cache size C / window W", "100 / 100 requests"],
    ["Episode", "One trace of 6 random phases, 3,000–7,999 requests each, disjoint key ranges"],
    ["Cache", "<code>MultiPolicyCache</code>: shared last-access, insertion time and aged frequency (half-life 10C)"],
    ["Seeds", `train ${c.seeds.train[0]}–${c.seeds.train[1]} · val ${c.seeds.val[0]}–${c.seeds.val[1]} · test ${c.seeds.test[0]}–${c.seeds.test[1]}`],
  ]);
  kv("f-dqn", [
    ["Network", `MLP ${c.obs_dim} → ${dq.hidden} → ${dq.hidden} → |𝒜|, ReLU`],
    ["Target", "Double DQN, Polyak τ = " + dq.tau],
    ["Loss / optimiser", `Huber · Adam, lr ${dq.lr}`],
    ["γ", dq.gamma],
    ["Replay / batch / warm-up", `${dq.buffer_size.toLocaleString()} / ${dq.batch_size} / ${dq.warmup.toLocaleString()}`],
    ["ε schedule", `${dq.eps_start} → ${dq.eps_end} linear over ${dq.eps_decay_steps.toLocaleString()} steps (60% of training)`],
    ["Reward scale", `× ${dq.reward_scale}`],
    ["Episodes", nEp("dqn_a3")],
  ]);
  const tb = c.tabular;
  kv("f-tab", [
    ["State", "best shadow policy × margin bin × own-hit-rate bin × previous action × Jaccard flag"],
    ["States", `${tb.states.a3} (3 actions) / ${tb.states.a4} (4 actions)`],
    ["Bins", `margin ${tb.margin_bins.join(", ")} · hit rate ${tb.hr_bins.join(", ")} · Jaccard < ${tb.jaccard_bin}`],
    ["α / γ", `${tb.alpha} / ${tb.gamma}`],
    ["ε schedule", "1.0 → 0.03 linear over 60% of training"],
    ["Episodes", nEp("qlearn_a3")],
  ]);
  const REG = {
    zipf: ["Zipf popularity, α ∈ [0.9, 1.2], universe 20C", "Frequency (LFU)"],
    drift: ["Hot set of 0.6–0.9C blocks that moves to fresh keys every 3–6C requests", "Recency (LRU)"],
    scan: ["Hot set of 0.5–0.7C blocks interleaved with one-shot sequential runs", "Scan resistance (aged LFU)"],
    stack: ["LRU stack-distance model, reuse probability 0.75–0.9", "Recency (LRU)"],
    loop: ["Cyclic sweep over 1.1–1.5C blocks plus a small hot set (30% of requests)", "MRU after warm-up"],
  };
  $("f-reg").innerHTML = "<tr><th>Regime</th><th>Generator</th><th>Favours</th></tr>" +
    c.regimes.map((r) => `<tr><td><code>${r}</code></td><td>${REG[r][0]}</td><td>${REG[r][1]}</td></tr>`).join("");
}

/* ---------------------------------------------------------------- wiring */
function renderAll() {
  renderOverall(); renderTimeline(); renderRegimes(); renderActions(); renderRecovery(); renderSizes(); renderTraining();
}
function setupControls() {
  const actOpts = [["a3", "3 (LRU/LFU/FIFO)"], ["a4", "4 (+MRU)"]];
  const sizeSeg = seg($("ov-size"), [[50, "50"], [100, "100"], [200, "200"]], 100, (v) => { OV.size = v; renderOverall(); });
  seg($("ov-act"), actOpts, "a3", (v) => {
    OV.act = v;
    if (v === "a4") { OV.size = 100; sizeSeg.set(100); sizeSeg.disable([50, 200]); } else sizeSeg.disable([]);
    renderOverall();
  });
  seg($("ov-metric"), Object.entries(METRICS).map(([k, v]) => [k, v.name]), "hit_rate", (v) => { OV.metric = v; renderOverall(); });

  const scn = $("tl-scn");
  Object.entries(D.config.scenarios).forEach(([k, v]) => scn.appendChild(h("option", { value: k }, v.regimes.join(" → "))));
  scn.value = TL.scn;
  scn.onchange = () => { TL.scn = scn.value; renderTimeline(); };
  seg($("tl-act"), actOpts, TL.act, (v) => { TL.act = v; renderTimeline(); });
  seg($("tl-agent"), [["DQN", "DQN"], ["Q-Learning", "Q-learning"], ["FTL", "FTL"]], TL.agent, (v) => { TL.agent = v; renderTimeline(); });
  tlOn = chips($("tl-chips"), LINE_METHODS, ["DQN", "ARC", "Fixed-LRU", "Belady"], renderTimeline);

  seg($("rg-act"), actOpts, RG.act, (v) => { RG.act = v; renderRegimes(); });
  seg($("ac-agent"), [["DQN", "DQN"], ["Q-Learning", "Q-learning"], ["FTL", "FTL"]], AC.agent, (v) => { AC.agent = v; renderActions(); });
  seg($("ac-act"), actOpts, AC.act, (v) => { AC.act = v; renderActions(); });

  seg($("rc-act"), actOpts, RC.act, (v) => { RC.act = v; renderRecovery(); });
  rcOn = chips($("rc-chips"), LINE_METHODS, ["DQN", "ARC", "Fixed-LRU", "FTL"], renderRecovery);
  szOn = chips($("sz-chips"), LINE_METHODS.filter((k) => k !== "Belady"), ["DQN", "Q-Learning", "ARC", "FTL", "Fixed-LRU"], renderSizes);

  const trc = $("tr-chips");
  Object.entries(RUNS).forEach(([k, r]) => {
    const b = h("button", { type: "button", class: "chip", "aria-pressed": "true" }, keyHtml(r.color, r.dash) + r.label);
    b.onclick = () => { trOn.has(k) ? trOn.delete(k) : trOn.add(k); b.setAttribute("aria-pressed", String(trOn.has(k))); renderTraining(); };
    trc.appendChild(b);
  });

  const root = document.documentElement, tbtn = $("theme");
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const sync = () => (tbtn.textContent = isDark() ? "Light mode" : "Dark mode");
  tbtn.onclick = () => { root.dataset.theme = isDark() ? "light" : "dark"; sync(); };
  sync();

  let t = null;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderAll, 120); });
}

fetch("data/results.json")
  .then((r) => r.json())
  .then((d) => {
    D = d;
    renderTiles(); setupControls(); renderAll(); renderStats(); renderFormulation();
  })
  .catch((e) => {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<p class="card">Could not load <code>data/results.json</code> (${e}). Serve this folder over HTTP, for example <code>python -m http.server -d docs</code>.</p>`);
  });
