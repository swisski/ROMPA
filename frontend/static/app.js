/* ROMP Onset Observatory — multi-model verification dashboard.
 * Plain vanilla JS; loaded after Plotly CDN.
 */
"use strict";

/* ------------------------------------------------------------------ *
 * Constants, palettes, theming
 * ------------------------------------------------------------------ */

const MODEL_PALETTE = [
  "#f0b264", // amber   (primary)
  "#6eb7ff", // sky
  "#b697e0", // violet
  "#86b97d", // green
  "#e87b85", // rose
  "#e8d06f", // sand
];

const ISO_DAY_PALETTE = ["#6eb7ff", "#86b97d", "#f0b264", "#e87b85"];

const FSS_COLORSCALE = [
  [0.0, "#11171f"],
  [0.4, "#223044"],
  [0.6, "#6eb7ff"],
  [1.0, "#f0b264"],
];

const PARAM_ORDER = [
  "wet_init",
  "wet_spell",
  "wet_threshold",
  "dry_spell",
  "dry_threshold",
  "dry_extent",
];

const PARAM_STEP = {
  wet_init: 1,
  wet_spell: 1,
  wet_threshold: 0.5,
  dry_spell: 1,
  dry_threshold: 0.5,
  dry_extent: 1,
};

const REGION_KEYS = ["lat_min", "lat_max", "lon_min", "lon_max"];

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(10,16,25,0.4)",
  font: { family: "IBM Plex Sans, system-ui, sans-serif", color: "#e4ddc9", size: 12 },
  margin: { l: 54, r: 20, t: 34, b: 42 },
  hoverlabel: {
    bgcolor: "#131e2a",
    bordercolor: "#f0b264",
    font: { family: "IBM Plex Mono, ui-monospace", color: "#e4ddc9", size: 11 },
  },
  xaxis: {
    gridcolor: "#1a2536",
    zerolinecolor: "#223044",
    linecolor: "#223044",
    tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#a8a291" },
  },
  yaxis: {
    gridcolor: "#1a2536",
    zerolinecolor: "#223044",
    linecolor: "#223044",
    tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#a8a291" },
  },
  legend: {
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 11 },
    orientation: "h",
    y: -0.22,
  },
};

const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
};

function mergeLayout(...parts) {
  // Shallow-deep merge for Plotly layouts. One level of nesting is enough.
  const out = {};
  for (const p of parts) {
    for (const k of Object.keys(p || {})) {
      const v = p[k];
      if (v && typeof v === "object" && !Array.isArray(v)) {
        out[k] = Object.assign({}, out[k] || {}, v);
      } else {
        out[k] = v;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * API helpers
 * ------------------------------------------------------------------ */

async function apiGet(path, params) {
  const url = params ? `${path}?${params}` : path;
  const res = await fetch(url);
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body && body.detail ? ` — ${body.detail}` : "";
    } catch (e) {
      /* ignore */
    }
    throw new Error(`${url}: HTTP ${res.status}${detail}`);
  }
  return res.json();
}

function qs(obj) {
  const parts = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined || v === null || v === "") continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return parts.join("&");
}

/* ------------------------------------------------------------------ *
 * Global state
 * ------------------------------------------------------------------ */

const state = {
  catalog: null,
  yearFrom: null,
  yearTo: null,
  isoYear: null,
  init: "auto",
  initIdx: null,
  models: [],           // array of {key,label,is_ensemble,n_members,on,primary}
  modelColor: {},       // key -> hex color
  params: {},
  region: { lat_min: "", lat_max: "", lon_min: "", lon_max: "" },
  busy: false,
  progressionShowDecomp: false,
  plotDivs: new Set(),
};

function selectedYears() {
  // Inclusive [from, to] within the primary model's available years.
  const avail = yearsForActiveSelection();
  if (!avail.length) return [];
  const a = state.yearFrom ?? avail[0];
  const b = state.yearTo ?? avail[avail.length - 1];
  const lo = Math.min(a, b), hi = Math.max(a, b);
  return avail.filter(y => y >= lo && y <= hi);
}

/* ------------------------------------------------------------------ *
 * Small UI helpers
 * ------------------------------------------------------------------ */

function $(id) { return document.getElementById(id); }

function setLoading(el, on) {
  if (!el) return;
  if (on) el.classList.add("is-loading");
  else el.classList.remove("is-loading");
}

function setStatus(text, kind /* "ok" | "err" | "" */) {
  const el = $("meta-status");
  if (!el) return;
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

function fmt(n, d) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function fmtE6(n, d = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return (n / 1e6).toFixed(d);
}

function fmtKm(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 100) return Math.round(n).toString();
  return n.toFixed(1);
}

// Hex (#rrggbb) -> "rgba(r,g,b,a)" string. Module-level so panels can
// share the same alpha-shaded color logic.
function rgbaHex(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16),
        g = parseInt(h.slice(2, 4), 16),
        b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// Format the bracketed uncertainty annotation for a season-scalar key.
// Prefers the 95% bootstrap CI (ci_lo / ci_hi); falls back to IQR
// (q25 / q75) when CIs are absent or degenerate (single year). Returns
// an empty string when neither is available.
function bracketAnnotation(season, key) {
  if (!season) return "";
  const ciLo = season[`${key}_ci_lo`];
  const ciHi = season[`${key}_ci_hi`];
  if (ciLo !== undefined && ciLo !== null && ciHi !== undefined && ciHi !== null) {
    // Single-year payloads collapse CI to a point mass; in that case
    // the bracket is uninformative — fall through to IQR (also null
    // in single-year, yielding no annotation).
    if (ciLo !== ciHi) {
      return ` <span class="cell-iqr" title="95% bootstrap CI on the median">`
        + `[${fmtE6(ciLo, 1)}–${fmtE6(ciHi, 1)}]</span>`;
    }
  }
  const q25 = season[`${key}_q25`], q75 = season[`${key}_q75`];
  if (q25 !== undefined && q25 !== null && q75 !== undefined && q75 !== null) {
    return ` <span class="cell-iqr" title="year-to-year IQR">`
      + `[${fmtE6(q25, 1)}–${fmtE6(q75, 1)}]</span>`;
  }
  return "";
}

function colorForModel(key) {
  if (state.modelColor[key]) return state.modelColor[key];
  const idx = Object.keys(state.modelColor).length % MODEL_PALETTE.length;
  state.modelColor[key] = MODEL_PALETTE[idx];
  return state.modelColor[key];
}

function primaryModelKey() {
  const on = state.models.filter(m => m.on);
  return on.length ? on[0].key : null;
}

function activeModelKeys() {
  return state.models.filter(m => m.on).map(m => m.key);
}

/* ------------------------------------------------------------------ *
 * buildParams — common query-string for every metric call
 * ------------------------------------------------------------------ */

function readParamsFromInputs() {
  const out = {};
  for (const p of PARAM_ORDER) {
    const el = document.querySelector(`#params input[data-key="${p}"]`);
    if (!el) continue;
    const v = el.value === "" ? null : Number(el.value);
    if (v !== null && !Number.isNaN(v)) out[p] = v;
  }
  return out;
}

function readRegionFromInputs() {
  const out = {};
  for (const k of REGION_KEYS) {
    const el = $(k);
    if (!el) continue;
    const raw = el.value;
    if (raw === "" || raw === null) continue;
    const v = Number(raw);
    if (!Number.isNaN(v)) out[k] = v;
  }
  return out;
}

function buildParams(extra) {
  // core: onset params + region (region only if provided)
  const p = readParamsFromInputs();
  const r = readRegionFromInputs();
  state.params = p;
  state.region = Object.assign({ lat_min: "", lat_max: "", lon_min: "", lon_max: "" }, r);
  return qs(Object.assign({}, p, r, extra || {}));
}

function metricParamsCSV(extra) {
  // years range for aggregated metrics (CRPS / FSS / displacement / progression / corp / compare)
  const years = selectedYears();
  const yearsArg = years.length
    ? (years.length === 1 ? { year: years[0] } : { years: `${years[0]}-${years[years.length - 1]}` })
    : {};
  return buildParams(Object.assign({}, yearsArg, extra || {}));
}

function isoParamsCSV(extra) {
  // single year for the isochrone hero
  const yr = state.isoYear ?? selectedYears().slice(-1)[0];
  return buildParams(Object.assign({ year: yr, init: state.init || "auto" }, extra || {}));
}

/* ------------------------------------------------------------------ *
 * Sidebar population
 * ------------------------------------------------------------------ */

function yearsForActiveSelection() {
  // Years available across the union of selected models, intersected with obs.
  // If nothing is selected yet, fall back to catalog.shared_years.
  const obsYears = new Set((state.catalog.obs && state.catalog.obs.years) || []);
  const active = state.models.filter(m => m.on);
  if (!active.length) return state.catalog.shared_years || [];
  const primary = active[0];
  const primaryYears = new Set(primary.years);
  const filtered = (state.catalog.shared_years || [])
    .filter(y => primaryYears.has(y) && obsYears.has(y));
  return filtered;
}

function fillYearOptions(sel, years, currentValue) {
  sel.innerHTML = "";
  if (!years.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "—"; opt.disabled = true;
    sel.appendChild(opt);
    return null;
  }
  for (const y of years) {
    const opt = document.createElement("option");
    opt.value = String(y); opt.textContent = String(y);
    sel.appendChild(opt);
  }
  const has = (v) => years.map(String).includes(String(v));
  if (currentValue !== null && currentValue !== undefined && has(currentValue)) {
    sel.value = String(currentValue);
    return Number(currentValue);
  }
  return Number(sel.value);
}

function populateYearSelect() {
  const years = yearsForActiveSelection();
  const fromSel = $("year_from"), toSel = $("year_to"), isoSel = $("iso_year");
  if (!years.length) {
    [fromSel, toSel, isoSel].forEach(s => {
      s.innerHTML = '<option value="" disabled>no overlap</option>';
    });
    state.yearFrom = state.yearTo = state.isoYear = null;
    refreshYearHint();
    return;
  }
  // Default range: last 5 years (or all if fewer). If the previously-
  // stashed state value is still in the current year set, honor it;
  // otherwise fall through to the default. A raw `??` misses the case
  // where the stashed value was set by a prior primary model and is no
  // longer valid for the new primary — explicit membership check instead.
  const validFrom = Number.isFinite(state.yearFrom) && years.includes(state.yearFrom);
  const validTo   = Number.isFinite(state.yearTo)   && years.includes(state.yearTo);
  const defFrom = validFrom ? state.yearFrom : years[Math.max(0, years.length - 5)];
  const defTo   = validTo   ? state.yearTo   : years[years.length - 1];
  state.yearFrom = fillYearOptions(fromSel, years, defFrom);
  state.yearTo   = fillYearOptions(toSel, years, defTo);
  // iso year: middle of currently-selected range, or middle of all years
  // when the selection is empty (avoids NaN).
  const sel = selectedYears();
  const pool = sel.length ? sel : years;
  const validIso = Number.isFinite(state.isoYear) && pool.includes(state.isoYear);
  const defIso = validIso ? state.isoYear : pool[Math.floor(pool.length / 2)];
  state.isoYear = fillYearOptions(isoSel, pool, defIso);
  refreshYearHint();
}

function refreshYearHint() {
  const yrs = selectedYears();
  const el = $("year-hint");
  if (el) {
    if (!yrs.length) el.textContent = "no overlap with primary model";
    else if (yrs.length === 1) el.textContent = "single year — no aggregation";
    else el.textContent = `aggregating ${yrs.length} years (${yrs[0]}–${yrs[yrs.length - 1]}); median + 95% bootstrap CI`;
  }

  // Low-n banner: warn loudly below 20 years (design-doc sampling guardrail).
  const banner = $("low-n-banner");
  const text = $("low-n-text");
  if (!banner || !text) return;
  if (yrs.length === 0 || yrs.length >= 20) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  if (yrs.length === 1) {
    text.textContent =
      "single year — every metric is a point estimate; bootstrap CIs " +
      "are degenerate (point mass) and not displayed.";
  } else {
    text.textContent =
      `${yrs.length} years is below the 20-year rule-of-thumb for stable ` +
      `onset-verification statistics. The 95% CI bands shown are real ` +
      `(percentile-method bootstrap on the year axis), but the underlying ` +
      `bootstrap distribution is itself uncertain at this N — treat narrow ` +
      `bands as provisional.`;
  }
}

function renderModelChips() {
  const container = $("models");
  container.innerHTML = "";
  state.models.forEach((m, idx) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.key = m.key;
    b.textContent = m.label + (m.is_ensemble ? ` · ${m.n_members}m` : "");
    b.title = m.is_ensemble
      ? `${m.label}: ${m.n_members}-member ensemble. Click to include/exclude. Leftmost active chip is the primary model (hero panels only use primary).`
      : `${m.label}: deterministic. Click to include/exclude. Leftmost active chip is the primary model.`;
    b.addEventListener("click", () => {
      const model = state.models.find(x => x.key === m.key);
      model.on = !model.on;
      applyChipClasses();
      // Year set depends on the (new) primary model's coverage.
      populateYearSelect();
      refresh();
    });
    container.appendChild(b);
    // pre-assign color by initial order
    colorForModel(m.key);
  });
  applyChipClasses();
}

function applyChipClasses() {
  // leftmost ON becomes primary; others ON become is-on; off = neither.
  const container = $("models");
  let primaryAssigned = false;
  const chips = Array.from(container.children);
  chips.forEach((chip) => {
    const key = chip.dataset.key;
    const model = state.models.find(x => x.key === key);
    chip.classList.remove("is-primary", "is-on");
    if (model && model.on) {
      if (!primaryAssigned) {
        chip.classList.add("is-primary");
        model.primary = true;
        primaryAssigned = true;
      } else {
        chip.classList.add("is-on");
        model.primary = false;
      }
      // tint chip with model color
      chip.style.setProperty("--chip-accent", colorForModel(key));
    } else {
      model.primary = false;
      chip.style.removeProperty("--chip-accent");
    }
  });
}

function renderParamInputs() {
  const container = $("params");
  container.innerHTML = "";
  const defaults = state.catalog.onset_defaults || {};
  const docs = state.catalog.onset_docs || {};
  for (const key of PARAM_ORDER) {
    const label = document.createElement("label");
    label.title = docs[key] || key;
    const span = document.createElement("span");
    span.textContent = key;
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(PARAM_STEP[key] || 0.1);
    input.dataset.key = key;
    if (defaults[key] !== undefined && defaults[key] !== null) {
      input.value = defaults[key];
    }
    input.addEventListener("change", () => {
      const btn = $("apply");
      btn.classList.add("is-dirty");
    });
    label.appendChild(span);
    label.appendChild(input);
    container.appendChild(label);
  }
}

function wireRegionInputs() {
  for (const k of REGION_KEYS) {
    const el = $(k);
    if (!el) continue;
    el.addEventListener("change", () => {
      const btn = $("apply");
      btn.classList.add("is-dirty");
    });
  }
}

/* ------------------------------------------------------------------ *
 * Init dropdown
 * ------------------------------------------------------------------ */

async function refreshInitOptions(primaryKey, year) {
  const sel = $("init");
  const currentVal = sel.value;
  sel.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "auto";
  auto.textContent = "auto-select overlap";
  sel.appendChild(auto);

  if (!primaryKey || !year) {
    sel.value = "auto";
    $("init-hint").textContent = "pick a primary model to load inits";
    return;
  }

  try {
    const data = await apiGet("/api/inits", qs({ model: primaryKey, year }));
    (data.inits || []).forEach((iso, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      // "2015-04-01T00:00:00" -> "#03 · 2015-04-01"
      const pretty = iso.slice(0, 10);
      opt.textContent = `#${String(i).padStart(2, "0")} · ${pretty}`;
      sel.appendChild(opt);
    });
    // preserve selection if still valid, else reset to auto
    if (currentVal && Array.from(sel.options).some(o => o.value === currentVal)) {
      sel.value = currentVal;
    } else {
      sel.value = "auto";
    }
    state.init = sel.value;
    $("init-hint").textContent = `auto picks whichever of ${data.n} init${data.n === 1 ? "" : "s"} overlaps obs`;
  } catch (e) {
    console.error("inits fetch failed", e);
    $("init-hint").textContent = "init list unavailable";
  }
}

/* ------------------------------------------------------------------ *
 * Summary table
 * ------------------------------------------------------------------ */

function renderSummaryTable(compare) {
  const tbl = $("summary-table");
  tbl.innerHTML = "";
  $("summary-year").textContent = compare && compare.year ? compare.year : "—";

  // Brackets in the IOE / SPS columns are 95% bootstrap CIs on the
  // median when multi-year, IQR when single-year is the only thing
  // available; the title tooltip names the active variant per cell.
  const heads = [
    {label: "Model"},
    {label: "yrs / mem"},
    {label: "IOE · 10⁶km²d", title: "median [95% bootstrap CI on median]"},
    {label: "SPS · 10⁶km²d", title: "median [95% bootstrap CI on median]"},
    {label: "Peak day", title: "DOY at which IOE(d) is maximised — when the front is most wrong"},
    {label: "CRPS · d"},
    {label: "Brier"},
    {label: "MCB / DSC"},
  ];
  for (const h of heads) {
    const d = document.createElement("div");
    d.className = "col-head";
    d.textContent = h.label;
    if (h.title) d.title = h.title;
    tbl.appendChild(d);
  }

  const rows = (compare && compare.rows) || [];
  // preserve the order compare gave us (primary first in compare.rows)
  rows.forEach((row, idx) => {
    const accent = colorForModel(row.model);
    const rowHead = document.createElement("div");
    rowHead.className = "row-head";
    rowHead.style.setProperty("--row-accent", accent);
    const rowHeadText = document.createElement("span");
    rowHeadText.className = "row-head-text";
    rowHeadText.textContent = row.label || row.model;
    rowHead.appendChild(rowHeadText);
    tbl.appendChild(rowHead);

    const members = document.createElement("div");
    members.className = "cell";
    const memTxt = row.is_ensemble ? `${row.n_members}m` : "det";
    members.textContent = (row.n_years && row.n_years > 1)
      ? `${row.n_years}y / ${memTxt}` : memTxt;
    tbl.appendChild(members);

    const season = (row.progression && row.progression.season) || {};
    // Bracket display shows the 95% bootstrap CI on the median when
    // available (multi-year), falling back to the IQR for single-year
    // panels where CIs are degenerate (point mass at the observed value).
    // Below the headline number, a small "X% misp" subline shows what
    // fraction of the season IOE is misplacement vs extent — the
    // diagnostic for "wrong size" (low misp%) vs "wrong place" (high misp%).
    const ioeCell = document.createElement("div");
    ioeCell.className = "cell" + (idx === 0 ? " primary" : "");
    const ioe = season.ioe_km2_day;
    let ioeHtml = fmtE6(ioe, 1) + bracketAnnotation(season, "ioe_km2_day");
    if (season.misp_frac !== null && season.misp_frac !== undefined) {
      const pct = Math.round(100 * season.misp_frac);
      const lo = season.misp_frac_ci_lo, hi = season.misp_frac_ci_hi;
      const ciTxt = (lo !== null && lo !== undefined && hi !== null && hi !== undefined && lo !== hi)
        ? ` [${Math.round(100*lo)}–${Math.round(100*hi)}]`
        : "";
      ioeHtml += ` <div class="cell-sub" title="fraction of season IOE that is misplacement (vs extent)">${pct}%${ciTxt} misp</div>`;
    }
    ioeCell.innerHTML = ioeHtml;
    tbl.appendChild(ioeCell);

    const spsCell = document.createElement("div");
    spsCell.className = "cell";
    const sps = season.sps_km2_day;
    if (!row.is_ensemble || sps === null || sps === undefined) {
      spsCell.textContent = "—";
    } else {
      spsCell.innerHTML = fmtE6(sps, 1) + bracketAnnotation(season, "sps_km2_day");
    }
    tbl.appendChild(spsCell);

    // Peak day of IOE — single-number diagnostic for "when is this
    // model's front most wrong?". A late-peaking model lags; an
    // early-peaking model leads.
    const peakCell = document.createElement("div");
    peakCell.className = "cell";
    const peak = (row.progression && row.progression.peak) || {};
    const peakDoy = peak.ioe_doy;
    if (peakDoy === null || peakDoy === undefined) {
      peakCell.textContent = "—";
    } else {
      const d = Math.round(peakDoy);
      const lo = peak.ioe_doy_ci_lo, hi = peak.ioe_doy_ci_hi;
      const ciTxt = (lo !== null && lo !== undefined && hi !== null && hi !== undefined && lo !== hi)
        ? ` <span class="cell-iqr" title="95% bootstrap CI on peak DOY">[${Math.round(lo)}–${Math.round(hi)}]</span>`
        : "";
      peakCell.innerHTML = `${d}${ciTxt}`;
    }
    tbl.appendChild(peakCell);

    const crpsCell = document.createElement("div");
    crpsCell.className = "cell";
    crpsCell.textContent = row.crps && row.crps.mean !== null ? fmt(row.crps.mean, 1) : "—";
    tbl.appendChild(crpsCell);

    const bsCell = document.createElement("div");
    bsCell.className = "cell";
    bsCell.textContent = row.corp && row.corp.bs !== null ? fmt(row.corp.bs, 3) : "—";
    tbl.appendChild(bsCell);

    const mdCell = document.createElement("div");
    mdCell.className = "cell";
    if (row.corp && row.corp.mcb !== null && row.corp.dsc !== null) {
      mdCell.textContent = `${fmt(row.corp.mcb, 3)} / ${fmt(row.corp.dsc, 3)}`;
    } else {
      mdCell.textContent = "—";
    }
    tbl.appendChild(mdCell);
  });

  if (!rows.length) {
    const msg = document.createElement("div");
    msg.className = "summary-placeholder";
    msg.textContent = "no rows returned";
    tbl.appendChild(msg);
  }
}

/* ------------------------------------------------------------------ *
 * Plot renderers
 * ------------------------------------------------------------------ */

function rememberPlot(div) { state.plotDivs.add(div); }

/* -- Isochrones (hero) -- */

function renderIsochrones(state_, iso, primaryLabel) {
  const traces = [];
  const MIN_SEG_VERTICES = 3;
  const days = (iso && iso.isochrones) || [];
  const labels = [];   // {x, y, text, color}

  // Background: a *discrete* per-iso-DOY band heatmap. Each cell is
  // shaded with the colour of the *first* iso DOY by which obs onset
  // has arrived there. This makes the "monsoon has visited" side of
  // each contour visually obvious — the cells inside the contour are
  // already shaded with that contour's colour. Without this shading
  // the reader has to learn the convention "south of a south-to-north
  // front means onset has arrived", which isn't visually self-evident.
  //
  // Ordering: the discrete heatmap is drawn first so contour lines
  // and "no data" overlays sit on top of it.
  if (state_ && state_.obs_onset && days.length) {
    const isoDayNums = days.map(e => e.day);
    // Discrete colorscale: one stop per iso DOY, mapping bin index i to
    // ISO_DAY_PALETTE[i]. Plotly heatmap interpolates between stops, so
    // we use stepwise stops [(t_i, c_i), (t_{i+1}, c_i)] to keep each
    // band a single colour.
    const N = isoDayNums.length;
    const colors = isoDayNums.map((_, i) => ISO_DAY_PALETTE[i % ISO_DAY_PALETTE.length]);
    const stops = [];
    for (let i = 0; i < N; i++) {
      const t0 = i / Math.max(1, N);
      const t1 = (i + 1) / Math.max(1, N);
      stops.push([t0, colors[i]]);
      stops.push([t1, colors[i]]);
    }
    // Per-cell bin: smallest i such that obs ≤ isoDayNums[i]. Cells whose
    // obs onset is later than every iso DOY plotted still fall into the
    // last bin — the band semantically reads as "obs onset is at or after
    // the second-to-last iso DOY, including cells visited even later in
    // the season." Without this extension, cells with late obs onset
    // would appear as unshaded "no data" gaps, which is misleading.
    const binZ = state_.obs_onset.values.map(row =>
      row.map(v => {
        if (v === null || (typeof v === "number" && Number.isNaN(v))) return null;
        for (let i = 0; i < N; i++) {
          if (v <= isoDayNums[i]) return i + 0.5;  // centre of bin i
        }
        return N - 0.5;   // visited but later than all iso DOYs → last bin
      })
    );
    traces.push({
      type: "heatmap",
      x: state_.obs_onset.lon,
      y: state_.obs_onset.lat,
      z: binZ,
      zmin: 0, zmax: N,
      colorscale: stops,
      showscale: false,
      opacity: 0.30,
      hoverinfo: "skip",
      name: "obs visited bands",
    });

    // "No data" cells: render NaN cells as a single distinct gray so the
    // reader can see where the obs field is missing — this resolves the
    // ambiguity of contours appearing to pass through "empty" regions
    // (they're really tracing the data/no-data boundary). Test for both
    // JSON null (FastAPI's default) and literal NaN (stdlib json with
    // allow_nan=True) so we catch either serialization path.
    const nanZ = state_.obs_onset.values.map(row =>
      row.map(v => (v === null || (typeof v === "number" && Number.isNaN(v)) ? 1 : null))
    );
    traces.push({
      type: "heatmap",
      x: state_.obs_onset.lon, y: state_.obs_onset.lat, z: nanZ,
      colorscale: [[0, "#3a4150"], [1, "#3a4150"]],
      showscale: false,
      opacity: 0.45,
      hoverinfo: "skip", name: "no data",
    });
  }

  days.forEach((entry, i) => {
    const color = ISO_DAY_PALETTE[i % ISO_DAY_PALETTE.length];
    const grp = `day-${entry.day}`;
    let bestSeg = null, bestLen = 0;

    // Forecast contours only. Observed onset is communicated via the
    // per-DOY band heatmap above; drawing observed lines on top of it
    // was visually redundant.
    //
    // Keep only the top-K longest segments. Real onset fields have
    // NaN cells scattered through the domain — each one creates a small
    // closed contour loop. The "real" front is captured by the 1–2
    // longest segments; the rest are detection-failure artefacts.
    const topK = (segs, k) => {
      if (!segs) return [];
      const filtered = segs.filter(s => s && s.length >= MIN_SEG_VERTICES);
      if (filtered.length <= k) return filtered;
      return filtered
        .slice()
        .sort((a, b) => b.length - a.length)
        .slice(0, k);
    };
    const pushSegs = (rawSegs, tag) => {
      const segs = topK(rawSegs, 4);
      let drawn = 0;
      (segs || []).forEach((seg) => {
        if (!seg || seg.length < MIN_SEG_VERTICES) return;
        const xs = seg.map(pt => pt[0]);
        const ys = seg.map(pt => pt[1]);
        traces.push({
          type: "scatter", mode: "lines+markers",
          x: xs, y: ys,
          line: { color, width: 2.2 },
          marker: {
            color: "rgba(0,0,0,0)",
            size: 5,
            line: { color, width: 1.4 },
            symbol: "circle",
          },
          opacity: 1.0,
          name: `DOY ${entry.day} · ${tag}`,
          legendgroup: grp, showlegend: drawn === 0,
          hovertemplate: `DOY ${entry.day} ${tag}<br>lon %{x:.2f}, lat %{y:.2f}<extra></extra>`,
        });
        drawn += 1;
        if (xs.length > bestLen) {
          bestLen = xs.length;
          bestSeg = { x: xs, y: ys };
        }
      });
    };
    pushSegs(entry.forecast, "fcst");

    if (bestSeg) {
      // Anchor at the easternmost point of the longest forecast contour.
      let idx = 0;
      for (let k = 1; k < bestSeg.x.length; k++) {
        if (bestSeg.x[k] > bestSeg.x[idx]) idx = k;
      }
      labels.push({ x: bestSeg.x[idx], y: bestSeg.y[idx],
                    text: `DOY ${entry.day}`, color });
    }
  });

  // Stagger label y-positions so close labels don't collide. Sort by y and
  // bump any label that's within MIN_DELTA degrees of the previous one.
  if (labels.length > 1) {
    const sorted = labels.slice().sort((a, b) => a.y - b.y);
    const MIN_DELTA = 1.5;  // degrees latitude
    for (let k = 1; k < sorted.length; k++) {
      const dy = sorted[k].y - sorted[k - 1].y;
      if (dy < MIN_DELTA) sorted[k].y = sorted[k - 1].y + MIN_DELTA;
    }
  }

  labels.forEach(l => {
    traces.push({
      type: "scatter", mode: "text",
      x: [l.x], y: [l.y], text: [l.text],
      textposition: "middle right",
      textfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: l.color },
      hoverinfo: "skip", showlegend: false,
    });
  });

  // Subtitle: onset criteria + iso-day spacing context.
  const params = (state.params && Object.keys(state.params).length)
    ? state.params
    : (state.catalog && state.catalog.onset_defaults) || {};
  const wi = params.wet_init, ws = params.wet_spell, wt = params.wet_threshold;
  const isoDays = (iso && iso.days) || [];
  const spacing = isoDays.length >= 2 ? (isoDays[1] - isoDays[0]) : null;
  const subtitleBits = [];
  if (wi != null && ws != null && wt != null) {
    subtitleBits.push(`onset = first ${ws}-day spell ≥ ${wi} mm/d, ∑ ≥ ${wt} mm (over the ${ws}-day spell)`);
  }
  if (spacing != null) {
    subtitleBits.push(`${isoDays.length} thresholds, ${spacing}-day spacing across the shared onset window`);
  }
  const subtitle = subtitleBits.join(" · ");

  const titleText = primaryLabel
    ? `${primaryLabel} · ${(iso && iso.year) || ""}<br>` +
      `<span style="font-family: IBM Plex Mono; font-size: 11px; color: #a8a291;">${subtitle}</span>`
    : "isochrones";

  const layout = mergeLayout(PLOT_LAYOUT, {
    title: {
      text: titleText,
      font: { family: "Fraunces, serif", color: "#e8e1cf", size: 16 },
      x: 0.01, xanchor: "left",
    },
    xaxis: { title: { text: "Longitude", font: { size: 11, color: "#a8a291" } } },
    yaxis: { title: { text: "Latitude",  font: { size: 11, color: "#a8a291" } } },
    margin: { l: 58, r: 90, t: 64, b: 54 },
    showlegend: true,
    hovermode: "closest",
  });
  layout.xaxis.scaleanchor = "y";

  const div = $("plot-isochrones");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Distance footer
  const foot = $("iso-distances");
  const parts = [];
  if (iso && iso.days && iso.days.length) {
    iso.days.forEach((d, i) => {
      const h = iso.hausdorff_km ? iso.hausdorff_km[i] : null;
      const f = iso.frechet_km ? iso.frechet_km[i] : null;
      const nf = iso.n_segments_fcst ? iso.n_segments_fcst[i] : 0;
      const no = iso.n_segments_obs ? iso.n_segments_obs[i] : 0;
      parts.push(`DOY ${d}: Hausdorff ${fmtKm(h)} km, Fréchet ${fmtKm(f)} km (${nf}f/${no}o segs)`);
    });
  }
  foot.textContent = parts.join("  ·  ") || "no isochrone days returned";
}

/* -- Progression curves (all models) -- */

function renderProgression(compare) {
  const traces = [];
  const rows = (compare && compare.rows) || [];
  const multiYear = (compare && compare.n_years > 1);

  const rgba = rgbaHex;   // local alias, hoisted helper at module scope

  rows.forEach((row) => {
    const color = colorForModel(row.model);
    const p = row.progression || {};
    const days = p.days || [];

    // For multi-year aggregates we shade the 95% bootstrap CI on the
    // median (the inferential quantity — overlap or non-overlap of two
    // models' bands answers "is this difference real?"). The IQR is
    // still in the JSON if needed; we drop it from the chart to avoid
    // stacking two shaded layers per model, which gets unreadable for
    // 4+ models.
    function bandKeys(prefix) {
      const lo = p[`${prefix}_ci_lo`], hi = p[`${prefix}_ci_hi`];
      if (multiYear && Array.isArray(lo) && Array.isArray(hi)) {
        return { lo, hi, label: "95% CI" };
      }
      // Single-year fallback: the JSON has CI = null and the chart
      // simply gets no band (single-year uncertainty is meaningless
      // without resampling — we don't fake one here).
      return null;
    }

    if (Array.isArray(p.ioe_km2)) {
      const band = bandKeys("ioe_km2");
      if (band) {
        traces.push({
          type: "scatter", x: days,
          y: band.hi.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          showlegend: false, hoverinfo: "skip",
        });
        traces.push({
          type: "scatter", x: days,
          y: band.lo.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          fill: "tonexty", fillcolor: rgba(color, 0.18),
          name: `${row.label} · ${band.label}`,
          showlegend: false, hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter",
        mode: multiYear ? "lines" : "lines+markers",
        x: days,
        y: p.ioe_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${row.label} · IOE${multiYear ? " (median + 95% CI)" : ""}`,
        line: { color, width: 2 },
        marker: { size: 4, color },
        connectgaps: false,
        hovertemplate: `${row.label}<br>DOY %{x}<br>IOE %{y:.2f}·10⁶ km²<extra></extra>`,
      });
    }
    if (Array.isArray(p.sps_km2) && p.sps_km2.some(v => v !== null && v !== undefined)) {
      const band = bandKeys("sps_km2");
      if (band) {
        traces.push({
          type: "scatter", x: days,
          y: band.hi.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          showlegend: false, hoverinfo: "skip",
        });
        traces.push({
          type: "scatter", x: days,
          y: band.lo.map(v => (v === null ? null : v / 1e6)),
          mode: "lines", line: { width: 0, color },
          fill: "tonexty", fillcolor: rgba(color, 0.12),
          showlegend: false, hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.sps_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${row.label} · SPS${multiYear ? " (median + 95% CI)" : ""}`,
        line: { color, width: 1.5, dash: "dot" },
        connectgaps: false,
        hovertemplate: `${row.label} SPS<br>DOY %{x}<br>%{y:.2f}·10⁶ km²<extra></extra>`,
      });
    }
  });

  // Optional decomposition for primary model
  if (state.progressionShowDecomp && rows.length) {
    const primary = rows[0];
    const color = colorForModel(primary.model);
    const p = primary.progression || {};
    const days = p.days || [];
    if (Array.isArray(p.extent_km2)) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.extent_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${primary.label} · extent`,
        line: { color, width: 1.2, dash: "dash" },
        opacity: 0.7,
        connectgaps: false,
      });
    }
    if (Array.isArray(p.misplacement_km2)) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: days,
        y: p.misplacement_km2.map(v => (v === null ? null : v / 1e6)),
        name: `${primary.label} · misplacement`,
        line: { color: "#e87b85", width: 1.2, dash: "dashdot" },
        opacity: 0.8,
        connectgaps: false,
      });
    }
  }

  // Peak-DOY annotation: faint vertical dashed line at the primary
  // model's IOE peak, with a CI band shaded behind it when multi-year.
  // We only draw this for the primary model — multi-model peak lines
  // would clutter the chart and the cross-model peak comparison
  // already lives in the bench summary table above.
  const shapes = [];
  const annotations = [];
  if (rows.length) {
    const primary = rows[0];
    const peak = (primary.progression && primary.progression.peak) || {};
    if (peak.ioe_doy !== null && peak.ioe_doy !== undefined) {
      const color = colorForModel(primary.model);
      const lo = peak.ioe_doy_ci_lo, hi = peak.ioe_doy_ci_hi;
      if (multiYear && lo !== null && lo !== undefined &&
          hi !== null && hi !== undefined && lo !== hi) {
        shapes.push({
          type: "rect", xref: "x", yref: "paper",
          x0: lo, x1: hi, y0: 0, y1: 1,
          fillcolor: rgba(color, 0.08), line: { width: 0 }, layer: "below",
        });
      }
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: peak.ioe_doy, x1: peak.ioe_doy, y0: 0, y1: 1,
        line: { color: rgba(color, 0.55), width: 1.2, dash: "dash" },
      });
      const ciTxt = (multiYear && lo !== null && lo !== undefined &&
                     hi !== null && hi !== undefined && lo !== hi)
        ? ` [${Math.round(lo)}–${Math.round(hi)}]` : "";
      annotations.push({
        x: peak.ioe_doy, y: 1, xref: "x", yref: "paper",
        text: `peak DOY ${Math.round(peak.ioe_doy)}${ciTxt}`,
        showarrow: false, xanchor: "left", yanchor: "top",
        font: { size: 10, color: rgba(color, 0.85) },
        bgcolor: "rgba(0,0,0,0)", xshift: 4,
      });
    }
  }

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Day of year", font: { size: 11, color: "#a8a291" } } },
    yaxis: { title: { text: "10⁶ km²", font: { size: 11, color: "#a8a291" } } },
    // Larger bottom margin + lower legend Y so the legend (can be 4-8
    // traces for multi-model + IQR bands) never overlaps the x-axis
    // ticks or the card's interpretation footer below.
    margin: { l: 56, r: 22, t: 22, b: 110 },
    legend: { orientation: "h", y: -0.35, yanchor: "top" },
    shapes, annotations,
  });

  const div = $("plot-progression");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Caption + toggle chip
  const capEl = $("progression-caption");
  capEl.innerHTML = "";

  // Chip toggle for decomposition
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "mini-chip" + (state.progressionShowDecomp ? " is-on" : "");
  toggle.textContent = state.progressionShowDecomp ? "decomp: on" : "decomp";
  toggle.addEventListener("click", () => {
    state.progressionShowDecomp = !state.progressionShowDecomp;
    renderProgression(compare);
  });
  capEl.appendChild(toggle);

  // Text summary
  const seasons = rows
    .map(r => r.progression && r.progression.season ? r.progression.season.ioe_km2_day : null)
    .filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  const txt = document.createElement("span");
  txt.className = "caption-text";
  // Read the bootstrap config off the primary model's progression block
  // (all models share the same aggregator settings within one /api/compare
  // call) so the caption can name the actual CI level used.
  const primaryProg = rows[0] && rows[0].progression;
  const ciLevel = primaryProg && primaryProg.ci_level
    ? Math.round(primaryProg.ci_level * 100) : 95;
  const nResamples = primaryProg && primaryProg.n_resamples
    ? primaryProg.n_resamples : null;
  const yrLabel = multiYear
    ? ` · ${compare.n_years} yrs (median line, ${ciLevel}% bootstrap CI shading${nResamples ? `, B=${nResamples}` : ""})`
    : "";
  if (seasons.length) {
    const lo = Math.min(...seasons) / 1e6;
    const hi = Math.max(...seasons) / 1e6;
    txt.textContent = `${rows.length} model${rows.length === 1 ? "" : "s"}${yrLabel} · season IOE range ${lo.toFixed(1)}–${hi.toFixed(1)} · 10⁶ km²·d`;
  } else {
    txt.textContent = `${rows.length} model${rows.length === 1 ? "" : "s"}${yrLabel}`;
  }
  capEl.appendChild(txt);
}

/* -- CORP reliability -- */

function renderCorp(corp, primaryLabel) {
  const traces = [];

  // Histogram of forecast probabilities (behind)
  const hist = corp.forecast_prob_histogram || [];
  if (hist.length) {
    const xs = hist.map((_, i) => (i + 0.5) / hist.length);
    traces.push({
      type: "bar",
      x: xs,
      y: hist,
      name: "freq",
      marker: { color: "rgba(168,162,145,0.25)", line: { width: 0 } },
      yaxis: "y2",
      hovertemplate: "p %{x:.2f}<br>count %{y}<extra></extra>",
      showlegend: false,
      width: hist.length ? 1 / hist.length * 0.95 : 0.05,
    });
  }

  // 45° perfect reliability
  traces.push({
    type: "scatter",
    mode: "lines",
    x: [0, 1],
    y: [0, 1],
    name: "perfect",
    line: { color: "#6a7689", width: 1, dash: "dash" },
    hoverinfo: "skip",
    showlegend: false,
  });

  // CORP calibration curve
  if (corp.curve) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: corp.curve.forecast_prob,
      y: corp.curve.calibrated_prob,
      name: "CORP",
      line: { color: "#f0b264", width: 2.5 },
      marker: { color: "#f0b264", size: 6 },
      hovertemplate: "p_fcst %{x:.2f}<br>p_cal %{y:.2f}<extra></extra>",
    });
  }

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: {
      title: { text: "forecast probability", font: { size: 11, color: "#a8a291" } },
      range: [0, 1],
    },
    yaxis: {
      title: { text: "observed / calibrated", font: { size: 11, color: "#a8a291" } },
      range: [0, 1],
    },
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      showticklabels: false,
      zeroline: false,
      showline: false,
    },
    margin: { l: 56, r: 30, t: 18, b: 48 },
    showlegend: false,
    bargap: 0.02,
  });

  const div = $("plot-corp");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Caption
  const tau = corp.tau;
  const n = corp.n;
  const res = corp.identity_residual;
  const resStr = (res === null || res === undefined || Number.isNaN(res))
    ? "—"
    : Number(res).toExponential(1);
  const neff = corp.n_effective;
  const mi = corp.moran_i;
  const effPart = (neff != null && n != null && neff < n)
    ? ` · N_eff = ${neff} (Moran I = ${mi != null ? mi.toFixed(2) : "—"})`
    : "";
  $("corp-caption").textContent =
    `τ = ${tau ?? "—"} · N = ${n ?? "—"}${effPart} · residual = ${resStr}`;

  // Breakdown
  const breakdown = $("corp-breakdown");
  breakdown.innerHTML = "";
  const items = [
    ["bs", corp.mean_score],
    ["mcb", corp.mcb],
    ["dsc", corp.dsc],
    ["unc", corp.unc],
  ];
  for (const [k, v] of items) {
    const it = document.createElement("div");
    it.className = "corp-item";
    const l = document.createElement("span");
    l.className = "corp-label";
    l.textContent = k;
    const val = document.createElement("span");
    val.className = "corp-val";
    val.textContent = fmt(v, 3);
    it.appendChild(l);
    it.appendChild(val);
    breakdown.appendChild(it);
  }
}

/* -- CRPS field -- */

function renderCrps(crps) {
  const f = crps.field || {};
  const traces = [{
    type: "heatmap",
    x: f.lon,
    y: f.lat,
    z: f.values,
    colorscale: "Magma",
    colorbar: {
      title: { text: "CRPS (days)", font: { size: 10, color: "#a8a291" } },
      thickness: 10,
      len: 0.75,
      tickfont: { size: 9, color: "#a8a291" },
    },
    hovertemplate: "lon %{x:.2f} · lat %{y:.2f}<br>CRPS %{z:.2f} d<extra></extra>",
  }];

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Longitude", font: { size: 11, color: "#a8a291" } }, scaleanchor: "y" },
    yaxis: { title: { text: "Latitude", font: { size: 11, color: "#a8a291" } } },
    margin: { l: 56, r: 80, t: 18, b: 48 },
  });
  layout.xaxis.scaleanchor = "y";

  const div = $("plot-crps");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  const yrs = (crps.n_years && crps.n_years > 1) ? ` · ${crps.n_years}-yr per-cell mean` : "";
  let extras = "";
  if (crps.median !== undefined && crps.median !== null) {
    extras = ` · median ${fmt(crps.median, 1)}d · IQR ${fmt(crps.q25, 1)}–${fmt(crps.q75, 1)}d`;
  }
  $("crps-caption").textContent =
    `mean ${fmt(crps.mean, 1)}d · max ${fmt(crps.max, 1)}d · ${crps.n_finite ?? "—"} cells${extras}${yrs}`;
}

/* -- Displacement dual-axis -- */

function renderDisplacement(disp) {
  const x = disp.thresholds || [];
  const km = disp.great_circle_km || [];
  const bias = (disp.area_bias_fraction || []).map(v => (v === null ? null : v * 100));

  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: km,
      name: "great-circle (km)",
      line: { color: "#6eb7ff", width: 2 },
      marker: { size: 5, color: "#6eb7ff" },
      yaxis: "y",
      hovertemplate: "DOY %{x}<br>shift %{y:.1f} km<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      x,
      y: bias,
      name: "area bias (%)",
      line: { color: "#f0b264", width: 2, dash: "dot" },
      marker: { size: 5, color: "#f0b264" },
      yaxis: "y2",
      hovertemplate: "DOY %{x}<br>area bias %{y:.1f}%<extra></extra>",
    },
  ];

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: { title: { text: "Threshold (DOY)", font: { size: 11, color: "#a8a291" } } },
    yaxis: {
      title: { text: "km", font: { size: 11, color: "#6eb7ff" } },
      tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#6eb7ff" },
    },
    yaxis2: {
      title: { text: "% area bias", font: { size: 11, color: "#f0b264" } },
      overlaying: "y",
      side: "right",
      showgrid: false,
      zeroline: false,
      tickfont: { family: "IBM Plex Mono, ui-monospace", size: 10, color: "#f0b264" },
    },
    legend: { orientation: "h", y: -0.28 },
    margin: { l: 56, r: 60, t: 18, b: 60 },
  });

  const div = $("plot-displacement");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);
}

/* -- FSS matrix -- */

function renderFss(fss) {
  const thresholds = fss.thresholds || [];
  const nbhds      = fss.neighborhoods || [];
  const z          = fss.fss || [];
  const baseRate   = fss.base_rate || [];
  const noSkill    = fss.fss_no_skill || baseRate;     // FSS_random = base rate
  const useful     = fss.fss_useful   || baseRate.map(b => b == null ? null : 0.5 + 0.5 * b);

  // Per-threshold useful-scale n* (the x-coordinate where FSS crosses the
  // useful-skill threshold). Single-year: array of floats / null on
  // `fss.useful_scale`. Multi-year: structured `fss.useful_scale.per_threshold`
  // with median + bootstrap CI per threshold.
  const usAgg = fss.useful_scale && fss.useful_scale.per_threshold
    ? fss.useful_scale.per_threshold : null;
  const usSingle = Array.isArray(fss.useful_scale) ? fss.useful_scale : null;
  function usForThreshold(i) {
    if (usAgg) {
      const e = usAgg[i] || {};
      return {n: e.useful_scale, lo: e.ci_lo, hi: e.ci_hi};
    }
    if (usSingle) {
      const v = usSingle[i];
      return {n: (v == null ? null : v), lo: null, hi: null};
    }
    return {n: null, lo: null, hi: null};
  }

  // colors graded across the threshold dimension (early → late onset)
  const lineColors = ["#6eb7ff", "#86b97d", "#f0b264", "#e87b85", "#b697e0",
                      "#e8d06f", "#a8d6f0"];
  const traces = [];

  thresholds.forEach((thr, i) => {
    const row = z[i] || [];
    const color = lineColors[i % lineColors.length];
    const ns = noSkill[i];
    const us = useful[i];
    const p  = baseRate[i];
    const labelP = p == null ? "—" : p.toFixed(2);

    // FSS line for this threshold
    traces.push({
      type: "scatter", mode: "lines+markers",
      x: nbhds, y: row,
      name: `DOY ${thr} · p=${labelP}`,
      legendgroup: `thr-${thr}`,
      line: { color, width: 2.4 },
      marker: { size: 6, color, line: { color, width: 1 } },
      hovertemplate:
        `<b>DOY ${thr}</b> at neighborhood %{x} cells<br>` +
        `FSS %{y:.3f}<br>` +
        `no-skill ${ns == null ? "—" : ns.toFixed(2)} · useful ${us == null ? "—" : us.toFixed(2)}` +
        `<extra></extra>`,
    });

    // Per-threshold useful reference (dotted, same color, low opacity)
    if (us != null) {
      traces.push({
        type: "scatter", mode: "lines",
        x: [nbhds[0], nbhds[nbhds.length - 1]], y: [us, us],
        line: { color, width: 1, dash: "dot" },
        opacity: 0.45,
        legendgroup: `thr-${thr}`, showlegend: false,
        hovertemplate: `useful threshold for DOY ${thr}: ${us.toFixed(2)}<extra></extra>`,
      });
    }
    // Per-threshold no-skill reference (very faint)
    if (ns != null) {
      traces.push({
        type: "scatter", mode: "lines",
        x: [nbhds[0], nbhds[nbhds.length - 1]], y: [ns, ns],
        line: { color, width: 1, dash: "dashdot" },
        opacity: 0.22,
        legendgroup: `thr-${thr}`, showlegend: false,
        hovertemplate: `no-skill (random=p) for DOY ${thr}: ${ns.toFixed(2)}<extra></extra>`,
      });
    }

    // Vertical marker at the useful scale n* — the smallest neighborhood
    // at which this threshold's FSS reaches the useful-skill cutoff.
    // Drawn from y=0 up to the useful threshold so the eye lands on the
    // crossing point. CI band shaded behind when multi-year.
    const usInfo = usForThreshold(i);
    if (usInfo.n != null && us != null) {
      if (usInfo.lo != null && usInfo.hi != null && usInfo.lo !== usInfo.hi) {
        traces.push({
          type: "scatter", mode: "lines",
          x: [usInfo.lo, usInfo.lo, usInfo.hi, usInfo.hi],
          y: [0, us, us, 0],
          fill: "toself",
          fillcolor: rgbaHex(color, 0.10),
          line: { width: 0 },
          legendgroup: `thr-${thr}`, showlegend: false,
          hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter", mode: "lines",
        x: [usInfo.n, usInfo.n], y: [0, us],
        line: { color, width: 1.4, dash: "dash" },
        opacity: 0.7,
        legendgroup: `thr-${thr}`, showlegend: false,
        hovertemplate: `useful scale for DOY ${thr}: n*=${usInfo.n.toFixed(1)} cells<extra></extra>`,
      });
    }
  });

  const layout = mergeLayout(PLOT_LAYOUT, {
    xaxis: {
      title: { text: "Neighborhood radius (cells)", font: { size: 11, color: "#a8a291" } },
      type: "linear", tickmode: "array", tickvals: nbhds, dtick: 1,
    },
    yaxis: {
      title: { text: "FSS", font: { size: 11, color: "#a8a291" } },
      range: [0, 1], dtick: 0.2,
    },
    annotations: [
      // global "useful" zone shading hint
      { xref: "paper", yref: "y", x: 0.005, y: 0.5,
        text: "↑ useful zone (≥ 0.5 + 0.5·p)",
        showarrow: false, xanchor: "left",
        font: { family: "IBM Plex Mono, ui-monospace", size: 9, color: "#a8a291" } },
    ],
    shapes: [
      { type: "rect", xref: "paper", yref: "y",
        x0: 0, x1: 1, y0: 0.5, y1: 1.0,
        fillcolor: "rgba(110, 183, 255, 0.04)", line: { width: 0 }, layer: "below" },
    ],
    legend: { orientation: "h", y: -0.28, font: { size: 11 } },
    margin: { l: 56, r: 22, t: 18, b: 70 },
  });

  const div = $("plot-fss");
  Plotly.react(div, traces, layout, PLOT_CONFIG);
  rememberPlot(div);

  // Per-threshold caption with useful-scale summary. Format depends on
  // whether multi-year (CI available) or single-year.
  const cap = $("fss-caption");
  if (cap) {
    const usParts = thresholds.map((thr, i) => {
      const u = usForThreshold(i);
      if (u.n == null) return `DOY ${thr}: no useful skill`;
      const ciTxt = (u.lo != null && u.hi != null && u.lo !== u.hi)
        ? ` [${u.lo.toFixed(1)}–${u.hi.toFixed(1)}]` : "";
      return `DOY ${thr}: n*=${u.n.toFixed(1)}${ciTxt}`;
    });
    const headline = thresholds.length
      ? `${thresholds.length} thresholds · base rates ${baseRate.filter(b => b != null).map(b => b.toFixed(2)).join(", ")}`
      : "";
    const usagg = fss.useful_scale && fss.useful_scale.bootstrap_method
      ? "" : "";  // (bootstrap method note can go here later if useful)
    cap.innerHTML = headline
      + (usParts.length
          ? ` <span class="caption-sub">· useful scales: ${usParts.join(" · ")}</span>`
          : "");
  }
}

/* ------------------------------------------------------------------ *
 * Orchestration
 * ------------------------------------------------------------------ */

async function loadCatalog() {
  setStatus("loading catalog…");
  const cat = await apiGet("/api/catalog");
  state.catalog = cat;

  // Seed models
  state.models = (cat.models || []).map((m, i) => ({
    key: m.key,
    label: m.label,
    is_ensemble: !!m.is_ensemble,
    n_members: m.n_members || 0,
    years: m.years || [],
    on: i < 2,      // auto-enable first two
    primary: i === 0,
  }));
  state.models.forEach(m => colorForModel(m.key));

  // Meta strings
  const nModels = state.models.length;
  $("meta-catalog").textContent = `${cat.obs?.label ?? "obs"} · ${nModels} model${nModels === 1 ? "" : "s"}`;

  const obsYears = cat.obs?.years || [];
  const yrLo = obsYears.length ? Math.min(...obsYears) : "—";
  const yrHi = obsYears.length ? Math.max(...obsYears) : "—";
  $("meta-obs").textContent = `IMD ${yrLo}–${yrHi} · ${obsYears.length} years`;
  const maskLabel = cat.land_mask || "off";
  $("meta-mask").textContent = maskLabel;

  // Colophon root
  const colo = $("colophon-root");
  if (colo) colo.textContent = cat.root || "—";

  populateYearSelect();
  renderModelChips();
  renderParamInputs();
  wireRegionInputs();

  setStatus("ready", "ok");
}

function markPanelsLoading(on) {
  [
    "plot-isochrones",
    "plot-progression",
    "plot-corp",
    "plot-crps",
    "plot-displacement",
    "plot-fss",
  ].forEach(id => setLoading($(id), on));
  setLoading($("summary-table"), on);
}

async function refresh() {
  if (state.busy) return;
  state.busy = true;
  $("apply").disabled = true;
  $("apply").classList.remove("is-dirty");

  try {
    // Pull the latest control values into state. Year-range and iso-year
    // are wired separately; here we just sync init.
    state.init = $("init").value || "auto";

    const primary = primaryModelKey();
    const actives = activeModelKeys();

    markPanelsLoading(true);

    if (!primary) {
      setStatus("pick at least one model", "err");
      $("summary-table").innerHTML = '<div class="summary-placeholder">select a model</div>';
      markPanelsLoading(false);
      return;
    }

    // Init options use the iso year (the only place a specific init makes sense)
    const isoYr = state.isoYear ?? selectedYears().slice(-1)[0];
    await refreshInitOptions(primary, isoYr);
    state.init = $("init").value || "auto";

    // Param strings
    const metricArgs   = (extra) => metricParamsCSV(Object.assign({ model: primary }, extra || {}));
    const isoArgs      = (extra) => isoParamsCSV(Object.assign({ model: primary }, extra || {}));
    const compareArgs  = metricParamsCSV({ models: actives.join(",") });
    const statePArgs   = isoParamsCSV({ model: primary });   // for hero state + isochrones

    setStatus("fetching…");

    // 1) compare (summary + progression) — multi-year aggregation
    const comparePromise = apiGet("/api/compare", compareArgs)
      .then(cmp => {
        renderSummaryTable(cmp);
        renderProgression(cmp);
      })
      .catch(err => {
        console.error("compare failed", err);
        $("summary-table").innerHTML =
          `<div class="summary-placeholder">compare failed: ${err.message}</div>`;
      })
      .finally(() => {
        setLoading($("summary-table"), false);
        setLoading($("plot-progression"), false);
      });

    // Hero panel: single-year state + isochrones for the iso year
    const statePromise = apiGet("/api/state", statePArgs)
      .then(s => apiGet("/api/metrics/isochrones", statePArgs).then(iso => ({ s, iso })))
      .then(({ s, iso }) => {
        const pLabel = (state.models.find(m => m.key === primary) || {}).label || primary;
        renderIsochrones(s, iso, pLabel);
      })
      .catch(err => {
        console.error("isochrones/state failed", err);
        try { Plotly.purge($("plot-isochrones")); } catch (e) { /* ignore */ }
        $("iso-distances").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-isochrones"), false));

    const crpsPromise = apiGet("/api/metrics/crps", metricArgs())
      .then(renderCrps)
      .catch(err => {
        console.error("crps failed", err);
        $("crps-caption").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-crps"), false));

    const dispPromise = apiGet("/api/metrics/displacement", metricArgs())
      .then(renderDisplacement)
      .catch(err => {
        console.error("displacement failed", err);
        try { Plotly.purge($("plot-displacement")); } catch (_) {}
        const div = $("plot-displacement");
        if (div) div.innerHTML =
          `<div class="plot-error">displacement: ${err.message}</div>`;
      })
      .finally(() => setLoading($("plot-displacement"), false));

    const corpPromise = apiGet("/api/metrics/corp", metricArgs())
      .then(renderCorp)
      .catch(err => {
        console.error("corp failed", err);
        $("corp-caption").textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-corp"), false));

    const fssPromise = apiGet("/api/metrics/fss",
        metricArgs({ neighborhoods: "1,3,5,7,9" }))
      .then(renderFss)
      .catch(err => {
        console.error("fss failed", err);
        try { Plotly.purge($("plot-fss")); } catch (_) {}
        const div = $("plot-fss");
        if (div) div.innerHTML =
          `<div class="plot-error">FSS: ${err.message}</div>`;
        const cap = $("fss-caption"); if (cap) cap.textContent = `error: ${err.message}`;
      })
      .finally(() => setLoading($("plot-fss"), false));

    await Promise.allSettled([
      comparePromise, statePromise, crpsPromise, dispPromise, corpPromise, fssPromise,
    ]);

    setStatus("ready", "ok");
  } catch (err) {
    console.error("refresh fatal", err);
    setStatus(`error: ${err.message}`, "err");
  } finally {
    markPanelsLoading(false);
    state.busy = false;
    $("apply").disabled = false;
  }
}

function cleanRegion(r) {
  const out = {};
  for (const k of REGION_KEYS) {
    if (r[k] !== "" && r[k] !== null && r[k] !== undefined) out[k] = r[k];
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Bindings
 * ------------------------------------------------------------------ */

function bindControls() {
  const readYearSel = (id) => {
    // Guard against the disabled "no overlap" placeholder (value = "") so
    // we don't poison state.yearFrom/yearTo with a literal 0 that would
    // block the default-recomputation path in populateYearSelect.
    const raw = $(id).value;
    if (raw === "" || raw === null) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  const onYearChange = () => {
    const from = readYearSel("year_from");
    const to   = readYearSel("year_to");
    if (from !== null) state.yearFrom = from;
    if (to   !== null) state.yearTo   = to;
    populateYearSelect();
    refresh();
  };
  $("year_from").addEventListener("change", onYearChange);
  $("year_to").addEventListener("change", onYearChange);

  document.querySelectorAll(".year-range-actions .mini-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const yrs = yearsForActiveSelection();
      if (!yrs.length) return;
      const tag = btn.dataset.yr;
      const toIdx = yrs.length - 1;
      let fromIdx = 0;
      if (tag === "single") fromIdx = toIdx;
      else if (tag === "all") fromIdx = 0;
      else fromIdx = Math.max(0, toIdx - (Number(tag) - 1));
      state.yearFrom = yrs[fromIdx]; state.yearTo = yrs[toIdx];
      populateYearSelect();
      refresh();
    });
  });

  $("iso_year").addEventListener("change", () => {
    const v = readYearSel("iso_year");
    if (v !== null) state.isoYear = v;
    refresh();
  });

  $("apply").addEventListener("click", () => { refresh(); });

  $("init").addEventListener("change", () => {
    state.init = $("init").value || "auto";
    refresh();
  });

  window.addEventListener("resize", () => {
    for (const d of state.plotDivs) {
      try { Plotly.Plots.resize(d); } catch (e) { /* ignore */ }
    }
  });
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

async function init() {
  try {
    await loadCatalog();
  } catch (err) {
    console.error("catalog failed", err);
    setStatus(`catalog error: ${err.message}`, "err");
    return;
  }
  bindControls();
  refresh();
}

document.addEventListener("DOMContentLoaded", init);
