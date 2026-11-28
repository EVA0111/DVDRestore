// EVA toolkit - renderer. All processing happens in the Python backend; this file is UI + state.
// Flow: Source (open / import disc / episodes) -> Analyze (all detectors) -> Repair (pipeline order, one Auto per group) -> Colour -> Crop & size -> Script -> Export.
"use strict";
const $ = (id) => document.getElementById(id);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const S = {
  opts: null, info: null, episodes: [], detected: [], checked: new Set(), current: 0,
  step: "source", analyzed: false, auto: {}, disc: null, titles: [], titleSel: 0, queue: {},
  scan: null, mode: "vfr", order: "tff", pal_mode: "auto", adv: {}, comb: "nnedi3", ai_model: "precise", stab: "off", stab_level: "normal", stab_pos: "default", repair_order: [], removed_steps: [],
  uni_crawl: 1, uni_rainbow: 1, uni_mpeg: 1, uni_halo: 1, uni_chroma: 1, uni_grain: 0, uni_detail: 1,
  crawl: "off", rainbow: "off", composite: "off",
  ghost: "off", ghost_strength: 1, ghost_taps: [], halo_rx: 0, halo_method: "ai", deblock: "off",
  ivtc_repair: "ai", chroma_unmix: "off", deblend: "off", aa: "off", rainbow_fill: false,
  sx: 0, sy: 0, bleed: "ai|normal", warp: 7, neutral: true,
  crop: { l: 0, r: 0, t: 0, b: 0 },
  grade: { black: 16, white: 235, shadows: 0, gamma: 1, contrast: 1, sat: 1, dark_sat: 0, temp: 0, tint: 0, gamut: "off",
           vibrance: 0, hue: 0, lift: [0, 0, 0], gamma_w: [0, 0, 0], gain_w: [0, 0, 0], curves: {}, luts: [] },
  layers: [], layerSel: -1, src: null,   // adjustment layers [{name, track, start, end (source frames), on, opacity, grade}]
  tracks: 3, trackOff: {},              // layer tracks L1 (bottom) .. Ln; trackOff[n] = hidden (eye)
  stage_order: ["colour", "size"], vfrAuto: null,
  levels: null, srcDir: "", autoStrength: 1, colourRef: [],
  autoUse: { levels: true, tone: true, white: true, black: true, fade: true, shadow: true, hue: true, gamut: true },
  aspect: "auto", aspect_custom: "", custom_w: 960, custom_h: 720, custom_lock: true,
  size: "keep", kernel: "spline36", kernel_b: 0.33, kernel_c: 0.33, tfix_strength: 2.0, encoder: "x264", crf: 16, out_dir: "", name: "Episode_{n:02d}",
  enc_rate: "q", enc_bitrate: 8000, enc_preset: "", enc_tune: "", enc_depth: 10, enc_chroma: "420", enc_keyint: 0, enc_extra: "",
  audio_codec: "copy", audio_bitrate: 192,
  busy: false, view: "after", zoom: "Fit", split: 50, img: null, frame: null, total: 0,
  backend: "pytorch", repairnet: "on", env: null,
  ai_dirs: [], ai_files: [], auto_update: true,            // your AI model folders / single files (filter window: AI upscale & restore models)
  sharp: { luma: 0, chroma: 0, mode: "faithful", radius: "fine", lines: 0 },
  vs_filters: [],                       // Repair -> VapourSynth filters: [{type: "vs"|"jet", on, p: {call, args, ...}}]
  vs_filters_pos: "repairs",            // where they run: source | ivtc (instead of the app's) | after_ivtc | repairs | end
  vs_groups: [],                        // Repair -> VapourSynth filter groups (draggable, duplicable): see renderVsGroups
  splits: [], cuts: [],
  chain: null, chainView: null,                  // Repair Chain mode: the blocks in their order (null = Guided); the block "View" shows                          // Timeline: edit points and parts cut out (source frames)
  ranges: [], range_only: {}, rangeSel: null,   // frame ranges of Repair steps (vsgraph.splice_src); rangeSel = the one edited
  upscale: { models: [] },              // [{path, name, info, strength, on, engine: {...}}]
  script: null, scriptGen: "",          // script = the user's edited text (null = follow the settings)
};
// Preferences: kept between sessions and for every DVD. Picture settings (PICTURE below) are not: they belong to
// one episode of one source and start from the defaults for every new DVD.
const SAVED = ["auto_update", "temp_dir", "stage_order", "backend", "repairnet", "ai_model", "ai_dirs", "ai_files", "encoder", "crf", "name", "adv",
               "enc_rate", "enc_bitrate", "enc_preset", "enc_tune", "enc_depth", "enc_chroma", "enc_keyint", "enc_extra", "audio_codec", "audio_bitrate"];
// Picture settings: Repair, Colour (with its adjustment layers), Crop & size, and what Auto measured.
// Every episode has its own (S.epSet[k]); a new episode starts from the defaults.
const PICTURE = ["ranges", "range_only", "scan", "mode", "order", "pal_mode", "comb", "ivtc_repair", "chroma_unmix", "aa", "deblend", "crawl", "rainbow", "composite", "rainbow_fill",
                 "deblock", "ghost", "ghost_strength", "ghost_taps", "halo_rx", "halo_method", "sx", "sy", "bleed", "warp", "neutral",
                 "stab", "stab_level", "stab_pos", "repair_order", "removed_steps", "tfix_strength", "crop", "aspect", "aspect_custom", "size", "custom_w", "custom_h", "custom_lock",
                 "uni_crawl", "uni_rainbow", "uni_mpeg", "uni_halo", "uni_chroma", "uni_grain", "uni_detail",
                 "kernel", "kernel_b", "kernel_c", "sharp", "upscale", "gradeAll", "layers", "tracks", "trackOff", "auto", "levels", "autoStrength", "autoUse", "colourRef",
                 "vs_filters", "vs_filters_pos", "vs_groups", "splits", "cuts", "chain"];
const GRADE0 = { black: 16, white: 235, shadows: 0, gamma: 1, contrast: 1, sat: 1, dark_sat: 0, temp: 0, tint: 0, gamut: "off",
                 vibrance: 0, hue: 0, lift: [0, 0, 0], gamma_w: [0, 0, 0], gain_w: [0, 0, 0], curves: {}, luts: [] };
// S.grade = the grade being edited: the selected adjustment layer's, or the base grade (S.gradeAll).
S.gradeAll = S.grade; delete S.grade;
Object.defineProperty(S, "grade", {
  get() { const l = S.layers[S.layerSel]; return l ? l.grade : S.gradeAll; },
  set(v) { const l = S.layers[S.layerSel]; if (l) l.grade = v; else S.gradeAll = v; },
});
const sameGrade = (a, b) => JSON.stringify(a) === JSON.stringify(b);
S.epSet = [];                                    // per-episode picture settings (null = defaults)
const PICTURE0 = {};                             // the defaults, taken before any saved setting is read
for (const k of PICTURE) PICTURE0[k] = structuredClone(S[k] ?? null);
function pictureSnap() { const o = {}; for (const k of PICTURE) o[k] = structuredClone(S[k] ?? null); return o; }
function pictureLoad(o) {
  for (const k of PICTURE) S[k] = structuredClone(o[k] ?? PICTURE0[k]);
  S.layerSel = -1;
}
function pictureFresh() {                        // defaults, with what the source itself says (field order)
  const o = structuredClone(PICTURE0);
  if (S.info) { o.order = S.info.order; o.pal_mode = "auto"; o.scan = S.info.progressive ? "progressive" : "interlaced"; }
  return o;
}
function epSetAlign() { while (S.epSet.length < S.episodes.length) S.epSet.push(null); S.epSet.length = S.episodes.length; }
// "mode" is not here: CFR / VFR / 60p change the frames shown, and a VFR analysis must stop when you leave VFR
const NO_PREVIEW = new Set(["encoder", "crf", "out_dir", "name", "enc_rate", "enc_bitrate", "enc_preset", "enc_tune",
                            "enc_keyint", "enc_extra", "audio_codec", "audio_bitrate"]);
const NO_SCRIPT = new Set(["out_dir", "name"]);   // settings the preview does not show

// ---------------------------------------------------------------- helpers
const fps = () => (S.info ? S.info.fps_num / S.info.fps_den : 30000 / 1001);
// Variable-frame-rate sources (24p and 30p parts): the probe sends every frame's time (ms steps); times shown or
// typed use them. Constant-rate sources: frames / rate as before.
let FT = { path: null, t: null };
function frameTimes() {
  const i = S.info;
  if (!i || !i.frame_ms) return null;
  if (FT.path !== i.path || !FT.t) {
    const t = new Float64Array(i.frame_ms.length);
    let c = 0;
    i.frame_ms.forEach((d, k) => { c += d; t[k] = c; });
    const base = t[0];
    for (let k = 0; k < t.length; k++) t[k] = (t[k] - base) / 1000;
    FT = { path: i.path, t };
  }
  return FT.t;
}
function frameTime(f) {                          // seconds from the first frame (f may be fractional)
  const t = frameTimes();
  if (!t || t.length < 2) {                        // constant rate; clock jumps the export closes are added back
    let c = f / fps(), acc = 0;
    for (const [tj, d] of (S.info && S.info.tl_jumps) || []) if (c + acc > tj) acc += d;
    return c + acc;
  }
  if (f <= 0) return f * (t[1] - t[0]);
  const n = t.length - 1;
  if (f >= n) return t[n] + (f - n) * (t[n] - t[n - 1]);
  const k = Math.floor(f);
  return t[k] + (f - k) * (t[k + 1] - t[k]);
}
function timeFrame(s) {                          // the frame shown at time s
  const t = frameTimes();
  if (!t) {
    let c = s;
    for (const [tj, d] of (S.info && S.info.tl_jumps) || []) if (s > tj) c -= Math.min(d, s - tj);
    return Math.round(c * fps());
  }
  let lo = 0, hi = t.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (t[m] < s - 0.0005) lo = m + 1; else hi = m; }
  return lo;
}
function tc(frames) {
  const s = frameTime(frames);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = (s % 60).toFixed(3).padStart(6, "0");
  return `${h}:${String(m).padStart(2, "0")}:${sec}`;
}
function hmsms(s) {                             // h:mm:ss.sss of a number of seconds
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = (s % 60).toFixed(3).padStart(6, "0");
  return `${h}:${String(m).padStart(2, "0")}:${sec}`;
}
function hms(sec) {
  sec = Math.round(sec);
  return `${Math.floor(sec / 3600)}:${String(Math.floor((sec % 3600) / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
}
function parseTc(text) {
  let s = 0;
  for (const p of String(text).trim().split(":")) s = s * 60 + parseFloat(p || "0");
  return timeFrame(s);
}
const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
function dirname(p) { const i = Math.max(p.lastIndexOf("\\"), p.lastIndexOf("/")); return i > 0 ? p.slice(0, i) : p; }
function basename(p) { return p.slice(Math.max(p.lastIndexOf("\\"), p.lastIndexOf("/")) + 1); }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

function log(msg, cls) {
  const box = $("log"), line = el("div", cls, msg);
  box.appendChild(line);
  while (box.childNodes.length > 1000) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
function status(msg, cls) {
  $("statusText").textContent = msg;
  $("statusText").style.color = cls === "err" ? "var(--danger)" : cls === "ok" ? "var(--go)" : "";
}

async function rpc(method, params) {
  const r = await window.api.rpc(method, params);
  if (r.error) {
    if (r.error !== "superseded" && r.error !== "Stopped") {
      log(`${method}: ${r.error}`, "err"); status(`${method}: ${r.error}`, "err");
    }
    throw new Error(r.error);
  }
  return r.result;
}

// state sent to the backend
function backendState(selected) {
  return {
    info: S.info && S.info.frame_ms ? { ...S.info, frame_ms: null } : S.info, episodes: S.episodes, selected,
    scan: S.scan || null, mode: S.mode, order: S.order, pal_mode: S.pal_mode, adv: S.adv, comb: S.comb, ai_model: S.ai_model, stab: S.stab, stab_level: S.stab_level, stab_pos: S.stab_pos || "default", repair_order: repairOrder(), removed_steps: S.removed_steps || [], tfix_strength: S.tfix_strength,
    uni_crawl: S.uni_crawl, uni_rainbow: S.uni_rainbow, uni_mpeg: S.uni_mpeg, uni_halo: S.uni_halo, uni_chroma: S.uni_chroma, uni_grain: S.uni_grain, uni_detail: S.uni_detail,
    crawl: S.crawl, rainbow: S.rainbow, composite: S.composite,
    ghost: S.ghost, ghost_strength: S.ghost_strength, ghost_taps: S.ghost_taps, halo_rx: S.halo_rx, halo_method: S.halo_method, deblock: S.deblock,
    ivtc_repair: S.ivtc_repair, chroma_unmix: S.chroma_unmix, deblend: S.deblend, aa: S.aa, rainbow_fill: S.rainbow_fill,
    ranges: S.ranges || [], range_only: S.range_only || {},
    sx: S.sx, sy: S.sy, bleed: S.bleed, warp: S.warp, neutral: S.neutral, vs_filters: S.vs_filters, vs_filters_pos: S.vs_filters_pos, vs_groups: S.vs_groups,
    crop: S.crop, size: S.size, kernel: S.kernel, kernel_b: S.kernel_b, kernel_c: S.kernel_c, encoder: S.encoder, crf: S.crf,
    enc_rate: S.enc_rate, enc_bitrate: S.enc_bitrate, enc_preset: S.enc_preset, enc_tune: S.enc_tune, enc_depth: S.enc_depth,
    enc_chroma: S.enc_chroma, enc_keyint: S.enc_keyint, enc_extra: S.enc_extra, audio_codec: S.audio_codec, audio_bitrate: S.audio_bitrate,
    colour_ref: S.colourRef, grade: S.gradeAll, layers: S.layers.filter((l) => !l.pending).sort((p, q) => (p.track || 1) - (q.track || 1) || p.start - q.start)     // bottom track first (layers not on the Timeline yet: not sent)
      .map((l) => ({ start: l.start, end: l.end, on: l.on !== false && !S.trackOff[l.track || 1], opacity: l.opacity ?? 1, grade: l.grade })),
    aspect: S.aspect, aspect_custom: S.aspect_custom,
    custom_w: S.custom_w, custom_h: S.custom_h, custom_lock: S.custom_lock,
    out_dir: S.out_dir, name: S.name, temp_dir: S.temp_dir || "", backend: S.backend, repairnet: S.repairnet, sharp: S.sharp, upscale: S.upscale,
    // Chain mode is the Guided settings laid out as a chain: the earlier separate block chain is never sent any more
    script: S.script, cuts: S.cuts || [], chain: null, chain_view: null,
    script_line: S.script ? S.scriptLine || null : null,
  };
}
function saveSettings() {
  if (window.__selftest) return;                 // automated UI checks must never change the user's settings
  const s = {};
  for (const k of SAVED) s[k] = S[k];
  window.api.rpc("save_settings", { settings: s });
}


// ---------------------------------------------------------------- auto-save and reload
// Auto-save (View menu, on by default): the picture settings of the source open - every episode's own settings, the layers, ranges,
// filters, cuts - are written to this app's storage a moment after every change, and come back when the same file is opened again
// (also after a reload or a crash). Off: settings start from the defaults for every source, as before.
const AUTOSAVE = { on: true, last: "", timer: 0 };
try { AUTOSAVE.on = localStorage.getItem("autoSave") !== "0"; } catch { /* no storage */ }
const asKey = (p) => "autosave:" + String(p).replace(/\\/g, "/").toLowerCase();
function autosaveMark() { const m = $("autosaveMark"); if (m) { m.textContent = AUTOSAVE.on ? "on" : "off"; m.style.color = AUTOSAVE.on ? "var(--ok, #4fb07e)" : ""; } }
function autosaveToggle() {
  AUTOSAVE.on = !AUTOSAVE.on; autosaveMark();
  try { localStorage.setItem("autoSave", AUTOSAVE.on ? "1" : "0"); } catch { /* no storage */ }
  if (AUTOSAVE.on) { AUTOSAVE.last = ""; autosaveTick(); }
  status(AUTOSAVE.on ? "Auto-save is on: your settings for this source are kept and come back when you open it again." : "Auto-save is off: settings are not kept between sessions.", "ok");
}
function autosaveTick() {
  if (!AUTOSAVE.on || !S.info || window.__selftest) return;
  try {
    epSetAlign();
    const cur = pictureSnap(), eps = S.epSet.map((x, k) => (k === S.current ? cur : x));
    const data = { path: S.info.path, frames: S.info.frames, episodes: S.episodes, checked: [...S.checked], current: S.current, eps, t: Date.now() };
    const json = JSON.stringify(data);
    if (json === AUTOSAVE.last) return;
    AUTOSAVE.last = json;
    const key = asKey(S.info.path);
    localStorage.setItem(key, json);
    let idx = JSON.parse(localStorage.getItem("autosaveIndex") || "[]").filter((k) => k !== key);   // the 15 most recent sources
    idx.unshift(key);
    for (const old of idx.splice(15)) localStorage.removeItem(old);
    localStorage.setItem("autosaveIndex", JSON.stringify(idx));
  } catch { /* storage full or unavailable: nothing is lost that was not already unsaved */ }
}
function autosaveRestore(path, frames) {           // called while a source opens; true = settings came back
  if (!AUTOSAVE.on) return false;
  let d; try { d = JSON.parse(localStorage.getItem(asKey(path)) || "null"); } catch { return false; }
  if (!d || !Array.isArray(d.eps) || !d.eps.length || (d.frames && frames && d.frames !== frames)) return false;
  S.episodes = d.episodes.map((e) => e.slice()); S.epSet = d.eps.map((x) => (x ? structuredClone(x) : null));
  S.checked = new Set(d.checked || S.episodes.map((_, k) => k)); S.current = Math.min(d.current || 0, S.episodes.length - 1);
  pictureLoad(S.epSet[S.current] || pictureFresh());
  AUTOSAVE.last = "";
  return true;
}
function autosaveForget() {
  if (!S.info) return;
  try { localStorage.removeItem(asKey(S.info.path)); } catch { /* none */ }
  status("The saved settings of this source are forgotten (what you see now stays until you close it).", "ok");
}
function reloadApp() {
  autosaveTick();
  if (S.info && !AUTOSAVE.on && !confirm("Auto-save is off: reloading drops the settings you changed on this source.\nReload anyway?")) return;
  try { if (S.info) localStorage.setItem("reopenAfterReload", JSON.stringify({ path: S.info.path, current: S.current })); } catch { /* none */ }
  location.reload();
}
setInterval(autosaveTick, 2000);
window.addEventListener("beforeunload", autosaveTick);
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "r") || e.key === "F5") { e.preventDefault(); reloadApp(); }
});

// ---------------------------------------------------------------- undo / redo
// Every change of the picture settings ends in schedulePreview(); half a second after the last one the state is
// recorded, so a slider drag or a typed number is one step. One history per source and episode.
const HIST = { stacks: {}, timer: 0, busy: false };
function histStack() {
  const k = `${S.info ? S.info.path : ""}#${S.current || 0}`;
  return HIST.stacks[k] || (HIST.stacks[k] = { undo: [], redo: [], cur: null });
}
function histArm() {
  if (HIST.busy || !S.info) return;
  clearTimeout(HIST.timer);
  HIST.timer = setTimeout(histCommit, 500);
}
function histCommit() {
  clearTimeout(HIST.timer);
  if (!S.info) return;
  const h = histStack(), now = JSON.stringify(pictureSnap());
  if (h.cur === null) { h.cur = now; histButtons(); return; }      // the state the episode opened with
  if (now === h.cur) return;
  h.undo.push(h.cur); if (h.undo.length > 300) h.undo.shift();
  h.redo = []; h.cur = now;
  histButtons();
}
function histGo(dir) {
  if (!S.info) return;
  histCommit();
  const h = histStack(), from = dir < 0 ? h.undo : h.redo, to = dir < 0 ? h.redo : h.undo;
  if (!from.length) { status(dir < 0 ? "Nothing to undo" : "Nothing to redo"); return; }
  to.push(h.cur); h.cur = from.pop();
  HIST.busy = true;
  try {
    togglePlay(false);
    const keep = S.layerSel;
    pictureLoad(JSON.parse(h.cur)); S.layerSel = keep;
    sync(); schedulePreview(); scheduleScript();
    if (typeof syncColourUi === "function") syncColourUi();
    trimTracks(); renderLayerTrack();
  } finally { HIST.busy = false; }
  histButtons();
  status(dir < 0 ? `Undone (${h.undo.length} more)` : `Redone (${h.redo.length} more)`);
}
function histButtons() {
  const h = S.info ? histStack() : { undo: [], redo: [] };
  $("btnUndo").disabled = !h.undo.length; $("btnRedo").disabled = !h.redo.length;
  $("btnUndo").title = `Undo (Ctrl+Z)${h.undo.length ? " - " + h.undo.length + " step(s)" : ""}`;
  $("btnRedo").title = `Redo (Ctrl+Y / Ctrl+Shift+Z)${h.redo.length ? " - " + h.redo.length + " step(s)" : ""}`;
}

// ---------------------------------------------------------------- workflows
// A workflow = the picture settings of an episode (Repair incl. crop & size, Colour, your filter groups and AI models) in
// a .json file you can keep, send to someone and load onto another source. The field order is the source's own.
const WF_SKIP = new Set(["order", "scan", "auto", "levels"]);   // field order and measurements belong to their own source
async function saveWorkflow() {
  if (!S.info) { status("Open a source first: a workflow is the settings of the episode shown.", "err"); return; }
  histCommit();
  const settings = {};
  for (const [k, v] of Object.entries(pictureSnap())) if (!WF_SKIP.has(k)) settings[k] = v;
  const name = (S.info.path.split(/[\\/]/).pop() || "workflow").replace(/\.[^.]+$/, "");
  const wf = { app: "EVA toolkit", kind: "workflow", version: 1, saved_with: document.querySelector(".brand .ver")?.textContent || "",
               created: new Date().toISOString(), name, source: S.info.path.split(/[\\/]/).pop(), settings };
  const p = await window.api.saveText(`${name}.evatoolkit.json`, JSON.stringify(wf, null, 1),
                                      [{ name: "EVA toolkit workflow", extensions: ["json"] }]);
  if (p) { log(`Workflow saved: ${p}`, "ok"); status(`Workflow saved: ${p}`, "ok"); }
}
async function loadWorkflow() {
  if (!S.info) { status("Open a source first: a workflow is applied to its episodes.", "err"); return; }
  const r = await window.api.openText([{ name: "EVA toolkit workflow", extensions: ["json"] }]);
  if (!r) return;
  let wf;
  try { wf = JSON.parse(r.text); } catch { status("Not a workflow file (not JSON).", "err"); return; }
  applyWorkflow(wf, r.path);
}
function applyWorkflow(wf, from) {
  const r = { path: from };
  const st = wf && wf.kind === "workflow" && wf.settings;
  if (!st || typeof st !== "object") { status("Not a EVA toolkit workflow file.", "err"); return; }
  const known = Object.keys(st).filter((k) => PICTURE.includes(k) && !WF_SKIP.has(k));
  if (!known.length) { status("This workflow holds no settings this version knows.", "err"); return; }
  const all = S.episodes.length > 1 && confirm(`Apply the workflow "${wf.name || ""}" to all ${S.episodes.length} episodes?\nOK = all episodes, Cancel = only episode ${S.current + 1}.`);
  histCommit();
  const apply = (o) => { const x = structuredClone(o || pictureFresh()); for (const k of known) x[k] = structuredClone(st[k]); return x; };
  epSetAlign();
  if (all) S.epSet = S.episodes.map((_, k) => apply(k === S.current ? pictureSnap() : S.epSet[k]));
  pictureLoad(apply(pictureSnap()));
  sync(); schedulePreview(); scheduleScript();
  if (typeof syncColourUi === "function") syncColourUi();
  renderLayerTrack();
  const missing = (S.vs_groups || []).flatMap((g) => g.filters).filter((f) => f.type === "ai" && f.p && f.p.args && f.p.args.model).length;
  log(`Workflow loaded from ${r.path}: ${known.length} settings${all ? " on all episodes" : ` on episode ${S.current + 1}`}` +
      (missing ? " (AI model filters point to model files on the PC it was saved on - check their paths)" : ""), "ok");
  status(`Workflow "${wf.name || r.path}" applied`, "ok");
}

// ---- Colour step panel under the preview (as Resolve's Colour page): Primaries = wheels + light + colour side by
// side; Curves / Layers / LUTs / More on their own page; the scopes always on the right
function initGradeTabs() {
  if (!$("gradeTabs")) return;                        // the Colour controls are in the Inspector now (workspace)
  const nav = $("gradeTabs"), pages = [...document.querySelectorAll("#gradeDock .gd-page")];
  const names = { primaries: "Primaries", curves: "Curves", layers: "Layers", luts: "LUTs", more: "More" };
  const tips = { primaries: "Colour wheels, light, colour, hue / shadow colour and curves", layers: "Adjustment layers: grades for a part of the video",
                 luts: "LUTs (.cube)" };
  let cur = "primaries";
  try { cur = localStorage.getItem("gradePage") || cur; } catch { /* no storage */ }
  const show = (id) => {
    for (const pg of pages) pg.classList.toggle("on", pg.dataset.page === id);
    for (const b of nav.children) b.classList.toggle("on", b.dataset.v === id);
    try { localStorage.setItem("gradePage", id); } catch { /* no storage */ }
    setTimeout(() => { try { syncColourUi(); } catch { /* not ready */ } }, 0);
  };
  pages.forEach((pg, i) => {
    const b = el("button"); b.dataset.v = pg.dataset.page; b.onclick = () => show(pg.dataset.page);
    b.append(el("kbd", null, String(i + 1)), el("span", null, names[pg.dataset.page] || pg.dataset.page));
    b.title = `${tips[pg.dataset.page] || ""} (key ${i + 1})`;
    nav.appendChild(b);
  });
  window.addEventListener("keydown", (e) => {        // 1-5 switch the page while in Colour (not while typing)
    if (S.step !== "colour" || e.ctrlKey || e.altKey || e.metaKey) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "range") return;
    const k = parseInt(e.key, 10);
    if (k >= 1 && k <= pages.length) { e.preventDefault(); show(pages[k - 1].dataset.page); }
  });
  show(pages.some((pg) => pg.dataset.page === cur) ? cur : "primaries");
  // collapse: only this bar stays, the picture gets the height (remembered; key H in Colour)
  const dock = $("gradeDock"), tog = el("button", "gd-collapse");
  const setCollapsed = (on) => {
    dock.classList.toggle("collapsed", on);
    document.body.classList.toggle("grade-collapsed", on);     // Auto correction + Scopes stack in one column
    tog.textContent = on ? "▴ Show panel" : "▾ Hide panel";
    tog.title = (on ? "Show the grading controls" : "Hide the grading controls - a bigger picture") + " (H)";
    try { localStorage.setItem("gradeCollapsed", on ? "1" : ""); } catch { /* no storage */ }
    setTimeout(layoutImage, 0);
  };
  tog.onclick = () => setCollapsed(!dock.classList.contains("collapsed"));
  nav.parentNode.appendChild(tog);
  for (const b of nav.children) b.addEventListener("click", () => { if (dock.classList.contains("collapsed")) setCollapsed(false); });
  let start = false;
  try { start = localStorage.getItem("gradeCollapsed") === "1"; } catch { /* no storage */ }
  setCollapsed(start);
  window.addEventListener("keydown", (e) => {
    if (S.step !== "colour" || e.ctrlKey || e.altKey || e.metaKey || (e.key !== "h" && e.key !== "H")) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "range") return;
    e.preventDefault(); setCollapsed(!dock.classList.contains("collapsed"));
  });
}

// ---------------------------------------------------------------- generic bound controls
// data-bind="key" (or "crop.l") on a select, .seg, .stepper, checkbox or text input.
function rangeEdited(key) {                      // the range whose own settings this control edits now, or null
  if (!S.rangeSel) return null;
  const r = (S.ranges || []).find((x) => x.id === S.rangeSel);
  return r && (RANGE_KEYS[r.step] || []).includes(key.split(".")[0]) ? r : null;
}
function getv(key) {
  const r = rangeEdited(key);
  if (r) { const top = key.split(".")[0], rest = key.split(".").slice(1); const base = top in (r.settings || {}) ? r.settings[top] : S[top]; return rest.reduce((o, k) => o[k], base); }
  return key.split(".").reduce((o, k) => o[k], S);
}
function setv(key, v) {
  const r = rangeEdited(key);
  if (r) {
    const ks = key.split("."), top = ks[0];
    r.settings = r.settings || {};
    if (ks.length === 1) r.settings[top] = v;
    else { if (!(top in r.settings)) r.settings[top] = structuredClone(S[top]); const last = ks.pop(); ks.slice(1).reduce((o, k) => o[k], r.settings[top])[last] = v; }
    return;
  }
  const ks = key.split("."), last = ks.pop(); ks.reduce((o, k) => o[k], S)[last] = v;
}

function changed(key, v) {
  if (PLAY.on && !NO_PREVIEW.has(key)) togglePlay(false);
  setv(key, v);
  if (key === "autoStrength") { applyAuto(); return; }
  if (SAVED.includes(key)) saveSettings();
  sync();
  if (!NO_PREVIEW.has(key)) schedulePreview();
  if (!NO_SCRIPT.has(key)) scheduleScript();
}

function buildControls() {
  for (const sel of $$("select[data-opts]")) {
    const items = S.opts[sel.dataset.opts].map((o) => (Array.isArray(o) ? o : [o.key, o.label]));
    sel.innerHTML = "";
    for (const [v, label] of items) { const o = el("option", null, label); o.value = v; sel.appendChild(o); }
  }
  for (const st of $$(".stepper[data-bind]")) {
    const step = parseFloat(st.dataset.step), min = parseFloat(st.dataset.min), max = parseFloat(st.dataset.max);
    const dec = el("button", null, "−"), inc = el("button", null, "+"), inp = el("input");
    inp.type = "text"; inp.inputMode = "decimal";
    st.append(dec, inp);
    if (st.dataset.unit) st.appendChild(el("span", "unit", st.dataset.unit));
    st.appendChild(inc);
    const clamp = (v) => parseFloat(Math.max(min, Math.min(max, Math.round(v / step) * step)).toFixed(4));
    const key = st.dataset.bind;
    dec.onclick = () => changed(key, clamp(getv(key) - step));
    inc.onclick = () => changed(key, clamp(getv(key) + step));
    inp.onchange = () => { const v = parseFloat(inp.value.replace(",", ".")); changed(key, isNaN(v) ? getv(key) : clamp(v)); };
    inp.onkeydown = (e) => {
      if (e.key === "ArrowUp" || e.key === "ArrowDown") { e.preventDefault(); changed(key, clamp(getv(key) + (e.key === "ArrowUp" ? step : -step))); }
    };
    if (st.dataset.hint) st.parentElement.appendChild(el("div", "field-hint", st.dataset.hint));
  }
  for (const c of $$("[data-bind]")) {
    const key = c.dataset.bind;
    if (c.classList.contains("seg")) for (const b of c.children) b.onclick = () => changed(key, b.dataset.v);
    else if (c.tagName === "SELECT") c.onchange = () => changed(key, c.value);
    else if (c.type === "checkbox") c.onchange = () => changed(key, c.checked);
    else if (c.tagName === "INPUT") c.onchange = () => changed(key, c.value);
  }
  // "auto" / "edited" chip on every field that analysis can set
  for (const f of $$(".field[data-key]")) {
    const chip = el("span", "chip hidden");
    chip.onclick = () => { if (f.dataset.key in S.auto) changed(f.dataset.key, S.auto[f.dataset.key]); };
    f.querySelector("label").appendChild(chip);
  }
}

function showRule(rule) {
  if (rule.includes("|")) return rule.split("|").some(showRule);
  if (rule.includes("&")) return rule.split("&").every(showRule);
  if (rule === "multi") return S.episodes.length > 1;
  const m = rule.match(/^([\w.]+)([=!^])(.*)$/);
  const v = String(getv(m[1]));
  return m[2] === "=" ? v === m[3] : m[2] === "!" ? v !== m[3] : v.startsWith(m[3]);
}

// state -> every control
// RepairNet (Settings -> Neural models -> Halos + bleed): the AI halo removal and the AI bleed fix run as ONE
// RepairNet pass. The Repair step says so, so it is clear which model does the work.
const RN_BLEED = { "ai|normal": "AI RepairNet (best)", "ai|strong": "AI RepairNet - strong (same as normal: one pass)",
                   "ai|light": "AI RepairNet - light (60 %)" };
function repairNetLabels() {
  const rn = S.repairnet !== "off";
  const hb = document.querySelector('[data-bind="halo_method"] button[data-v="ai"]');
  if (hb) hb.textContent = rn ? "AI · RepairNet" : "AI";
  for (const o of document.querySelectorAll('select[data-bind="bleed"] option')) {
    if (!(o.value in RN_BLEED)) continue;
    if (!o.dataset.orig) o.dataset.orig = o.textContent;
    o.textContent = rn ? RN_BLEED[o.value] : o.dataset.orig;
  }
  const haloAi = S.ghost === "auto" && S.halo_method === "ai";
  const bleedAi = String(S.bleed || "").startsWith("ai|");
  const msg = haloAi && bleedAi
    ? "Halos and the colour bleed fix run together in one RepairNet pass (switch: Settings → Neural models → Halos + bleed)."
    : "Runs as RepairNet (switch: Settings → Neural models → Halos + bleed).";
  $("rnNoteHalo").textContent = msg;
  $("rnNoteColour").textContent = msg;
  $("rnNoteHalo").classList.toggle("hidden", !(rn && haloAi));
  $("rnNoteColour").classList.toggle("hidden", !(rn && bleedAi));
}

// ---------------------------------------------------------------- Repair -> VapourSynth filter groups
// Groups of your own VapourSynth plugin / vs-jetpack functions (EVA toolbox library: names, arguments, defaults,
// compatibility scan). Each group is dragged by its handle between the Repair groups: where it sits is where it runs
// (vsgraph.vs_filters at that point); a group before / right after Telecine can do the IVTC itself. ⧉ duplicates it.
// S.vs_groups = [{id, slot 0-6, own_ivtc, on, filters: [{type, on, label, doc, argspec, p: {call, args, ...}}]}]
let VSF_LIB = null, VSF_GROUPS = null;
const vsfOpen = {};                               // group id -> index of the filter whose settings are shown
// A filter group remembers the Repair group it follows (g.after = its data-gid, "" = before Telecine): the Repair
// groups after Composite artifacts can be dragged too (S.repair_order, vsgraph.repair_order), so numbers would shift.
const STATIC_IDS = ["telecine", "composite", "compression", "halos", "colour", "stab", "crop", "size", "sharpen"];   // old slot numbers
const REPAIR_STEPS = ["compression", "halos", "colour", "crop", "stab", "size", "sharpen"];   // movable, usual order
const AFTER_POINT = { "": "source", telecine: "after_ivtc", composite: "after_composite", compression: "after_cleanup",
  halos: "after_halos", colour: "repairs", stab: "after_stab", crop: "after_crop", size: "after_size", sharpen: "after_sharpen" };
const STEP_NAME = { telecine: "Telecine & interlacing", composite: "Composite artifacts", compression: "Compression", halos: "Halos",
  colour: "Colour alignment", stab: "the Temporal stabilizer", crop: "Crop", size: "Shape and size", sharpen: "Sharpen & lines" };
function sharpAfterSize() { const o = repairOrder(); return o.indexOf("sharpen") > o.indexOf("size"); }
function repairOrder() {                           // the movable steps in their order (older settings: from stab_pos)
  const o = Array.isArray(S.repair_order) ? [...new Set(S.repair_order)].filter((x) => REPAIR_STEPS.includes(x)) : [];
  if (o.length) { REPAIR_STEPS.forEach((x, i) => { if (!o.includes(x)) o.splice(Math.min(i, o.length), 0, x); }); return o; }
  const out = REPAIR_STEPS.filter((x) => x !== "stab");
  const a = { after_ivtc: null, after_cleanup: "compression", after_halos: "halos", after_size: "size", end: "sharpen" }[S.stab_pos] ?? "crop";
  out.splice(a === null ? 0 : out.indexOf(a) + 1, 0, "stab");
  return out;
}
const AFTER_NAME = new Proxy({}, { get: (_, a) => {
  if (a === "") return "before Telecine & interlacing";
  const o = repairOrder();
  if (a === o[o.length - 1]) return `at the very end (after ${STEP_NAME[a]})`;
  return STEP_NAME[a] ? `after ${STEP_NAME[a]}` : undefined;
} });
function vsfAfter(g) {                             // older groups: position number -> the group it follows
  if (g.after === undefined) { const k = +g.slot || 0; g.after = k <= 0 ? "" : STATIC_IDS[Math.min(k, 9) - 1]; }
  return g.after;
}
function vsfIvtcSpot(g) { const a = vsfAfter(g); return a === "" || a === "telecine"; }
function vsfPoint(g) {                            // -> the engine's point (vsgraph.VSF_POINTS)
  if (g.own_ivtc && vsfIvtcSpot(g)) return "ivtc";
  const a = vsfAfter(g), o = repairOrder();
  return a === o[o.length - 1] ? "end" : AFTER_POINT[a] || "repairs";
}
function vsfWhere(g) { return AFTER_NAME[vsfAfter(g)] || "after Colour alignment"; }
async function vsfLibrary() {
  if (!VSF_LIB) {
    const r = await rpc("block_library", {});
    VSF_GROUPS = [VSF_AI_GROUP, VSF_MY_GROUP, ...(r.groups || [])];
    VSF_MYDIR = r.my_dir || "";
    VSF_LIB = [...(await vsfAiModels()), VSF_CODE, ...(await vsfNodes()), ...(r.my || []), ...(r.vs || []), ...(r.jet || [])].filter((f) => f.status !== "error" || f.kind === "my");
  }
  return VSF_LIB;
}
// ---- your own AI models (core/aiupscale.py via blocks "ai"): every .onnx / .pth / .safetensors in your model folders
// ---- your own filters: Custom code, functions in my_filters\*.py, plugins in my_filters\plugins
let VSF_MYDIR = "";
const VSF_MY_GROUP = ["my", "Your own filters",
  "Custom code (a few lines of VapourSynth Python), your Python functions (my_filters\\*.py: every function taking the clip first) and plugins you added (my_filters\\plugins\\*.dll)."];
const VSF_CODE = { kind: "code", call: "code", name: "New custom filter", group: "my", status: "ok", shown: "Python",
  doc: "Write your own filter: a name, what it does, its settings and code (`clip` in, `clip` out). Save it with the disk icon on its card and it appears here, under Your own filters.",
  args: [{ name: "code", type: "Python" }], defaults: { code: "clip = clip" } };
async function vsfMyAdd() {
  const files = await window.api.openMyFilter();
  if (!files || !files.length) return;
  let r;
  try { r = await rpc("my_filter_add", { paths: files }); } catch { return; }
  log(`Added to your filters: ${r.added.join(", ")}${r.loaded.length ? " (plugins loaded: " + r.loaded.join(", ") + ")" : ""}`, "ok");
  VSF_LIB = null; $("vsfDlgRows").textContent = "";
  await vsfLibrary(); VSFB.cat = "my"; vsfBrowserRender();
}
function vsfMyBar() {
  const bar = $("vsfMyBar"); bar.textContent = "";
  bar.classList.toggle("hidden", VSFB.cat !== "my");
  if (VSFB.cat !== "my") return;
  const add = el("button", null, "+ Add file…"); add.onclick = vsfMyAdd;
  add.title = "A custom filter someone exported (.json), a .py file (its functions become filters) or a VapourSynth plugin .dll - copied into my_filters";
  const open = el("button", null, "Open my_filters folder"); open.onclick = () => VSF_MYDIR && window.api.showPath(VSF_MYDIR);
  const re = el("button", null, "Read again"); re.title = "After editing your .py files"; re.onclick = async () => { VSF_LIB = null; await vsfLibrary(); vsfBrowserRender(); };
  bar.append(add, open, re, el("span", "dim", "A function's first argument is the clip, the others are its settings; its docstring is the help."));
}
const VSF_AI_GROUP = ["ai", "AI upscale & restore models",
  "Your own models (VapourKit, chaiNNer, OpenModelDB): .onnx on TensorRT, .pth / .safetensors on PyTorch. Upscalers make the picture bigger - best in a group after the stabilizer, so the slow steps run at DVD size."];
const VSF_AI_ARGS = [{ name: "model", type: "file", opt: false }, { name: "strength", type: "float", opt: true },
                     { name: "colour_fix", type: "off / on", opt: true }];
async function vsfAiModels() {
  let r = { models: [] };
  try { r = await rpc("ai_models", { dirs: S.ai_dirs || [], files: S.ai_files || [] }); } catch { /* none */ }
  return (r.models || []).map((m) => {
    const ext = m.file.split(".").pop().toLowerCase();
    return { kind: "ai", call: "ai.model", shown: (ext === "onnx" ? "ONNX" : "PyTorch ." + ext) + (m.scale ? ` · ${m.scale}x` : ""),
      name: m.file.replace(/\.(onnx|pth|safetensors)$/i, ""), group: "ai", plugin: m.folder, status: "ok", path: m.path, mb: m.mb, scale: m.scale,
      doc: (m.scale === 1 ? "1x restore model (same size)" : m.scale ? `${m.scale}x upscaler` : "AI model") + ` · ${m.mb} MB · ${m.path}`,
      args: VSF_AI_ARGS, defaults: { model: m.path, strength: "1", colour_fix: "off" } };
  });
}
async function vsfAiReload() {
  saveSettings();
  if (!VSF_LIB) return;
  VSF_LIB = [...(await vsfAiModels()), ...VSF_LIB.filter((f) => f.kind !== "ai")];
  VSFB.cat = "ai"; vsfBrowserRender();
}
function vsfAiBar() {                              // above the table in the AI category: your folders / files
  const bar = $("vsfAiBar"); bar.textContent = "";
  bar.classList.toggle("hidden", VSFB.cat !== "ai");
  if (VSFB.cat !== "ai") return;
  const addDir = el("button", null, "+ Add model folder");
  addDir.onclick = async () => { const d = await window.api.openFolder("Folder with your AI models (.onnx / .pth / .safetensors)");
    if (d && !S.ai_dirs.includes(d)) { S.ai_dirs = [...S.ai_dirs, d]; vsfAiReload(); } };
  const addFile = el("button", null, "+ Add model file");
  addFile.onclick = async () => { const fs = await window.api.openModel();
    const add = (fs || []).filter((f) => !S.ai_files.includes(f)); if (add.length) { S.ai_files = [...S.ai_files, ...add]; vsfAiReload(); } };
  bar.append(addDir, addFile);
  const list = el("div", "vsf-ai-src");
  for (const [k, label] of [["ai_dirs", "folder"], ["ai_files", "file"]])
    for (const p of S[k] || []) {
      const chip = el("span", "vsf-ai-chip"); chip.title = `${label}: ${p}`;
      const x = el("button", "link", "✕"); x.title = "Remove from the list (the file stays)";
      x.onclick = () => { S[k] = S[k].filter((q) => q !== p); vsfAiReload(); };
      chip.append(el("span", null, p), x); list.appendChild(chip);
    }
  if (!(S.ai_dirs.length || S.ai_files.length))
    list.appendChild(el("span", "dim", "Add the folder that holds your models, or single model files."));
  bar.appendChild(list);
}
function vsgChanged() {
  for (const g of S.vs_groups) g.pos = vsfPoint(g);
  changed("vs_groups", S.vs_groups.map((g) => structuredClone(g)));
}
function vsfArgsOf(f) {                          // [{name, hint}] for the settings form
  if (f.kind === "code") return [{ name: "code", hint: "Python: change `clip`" }];
  if (f.kind === "ai") return [{ name: "model", hint: "model file (.onnx / .pth / .safetensors)" }, { name: "strength", hint: "0-1, default 1 (less = blend with a plain resize)" },
    { name: "colour_fix", hint: "off / on (keep the colours of the picture before the model)" }];
  if (f.kind === "vs") return (f.args || []).map((a) => ({ name: a.name, hint: !a.type.startsWith("vnode") ? a.type + (a.opt ? "" : " (required)")
    : a.auto ? "clip - made automatically (or @group / @prev / @self)" : "clip: @group, @prev or @self" + (a.opt ? "" : " (required)") }));
  return (f.args || []).map((a) => ({ name: a.name, hint: /VideoNode/.test(a.type || "") && a.default === "" ? "clip: group, prev or current"
    : a.default !== "" ? `default ${a.default}` : (a.required ? "required" : "") }));
}
function vsfNewGroup(after = "colour", filters = []) {
  return { id: "g" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), after, own_ivtc: false, on: true, filters };
}
function vsfMigrate() {                           // the earlier single list -> one group
  if (S.vs_filters && S.vs_filters.length && !(S.vs_groups || []).length) {
    const after = { source: "", ivtc: "", after_ivtc: "telecine", repairs: "colour", end: "sharpen" }[S.vs_filters_pos] ?? "colour";
    S.vs_groups = [Object.assign(vsfNewGroup(after, S.vs_filters.map((f) => structuredClone(f))), { own_ivtc: S.vs_filters_pos === "ivtc" })];
    S.vs_filters = [];
  }
  if (!Array.isArray(S.vs_groups)) S.vs_groups = [];
}
function vsfStaticGroups() {                      // the Repair step's own groups, in order (the drop slots between them)
  return [...document.querySelectorAll('[data-panel="adjust"] > .group:not(.vsf-group)')];   // incl. Crop, Shape and size, Sharpen
}

// your custom filters (VapourKit-style): saved from a filter card (name, what it does, code) as my_filters\nodes\*.json
function vsfNodeItems(list) {
  return (list || []).map((n) => ({ kind: "node", call: "node", name: n.name, group: "my", status: "ok", doc: n.doc, code: n.code, file: n.file, pickId: "node:" + n.name }));
}
async function vsfNodes() {
  try { return vsfNodeItems((await rpc("my_nodes", {})).nodes); } catch { return []; }
}
function vsfNodesSet(list) {                         // after a save / delete: the library follows
  if (VSF_LIB) VSF_LIB = [...VSF_LIB.filter((f) => f.kind !== "node"), ...vsfNodeItems(list)];
  if (!$("vsfDlg").classList.contains("hidden") && typeof pickBuild === "function") { pickBuild(); pickRender(); }
}
const VSF_NEW_CODE = "# Your own filter: its settings first, then the code. `clip` comes in, `clip` goes out.\n" +
  "radius     = 2          # a setting - change it here\n\n\n" +
  "clip = core.std.BoxBlur(clip, hradius=radius, vradius=radius)   # your VapourSynth code";
async function vsfNodeSave(g, f) {
  const name = (f.label || "").trim();
  if (!name || (name === "New filter" && !f.node)) { status("Give the filter its own name first (the field above the code)", "err"); return; }
  let r;
  try { r = await rpc("my_node_save", { name, doc: f.doc || "", code: (f.p.args && f.p.args.code) || "", old: f.node && f.node !== name ? f.node : "" }); }
  catch (e) { status("Not saved: " + e, "err"); return; }
  if (r.error) { status(r.error, "err"); return; }
  f.node = name; vsgChanged();
  vsfNodesSet(r.nodes);
  status(`Saved "${name}" to your filters (library › Your own filters)`, "ok");
}
async function vsfNodeDelete(g, f) {
  if (!f.node || !confirm(`Delete "${f.node}" from your filters in the library?\n(The filter in this group stays.)`)) return;
  const r = await rpc("my_node_delete", { name: f.node });
  delete f.node; vsgChanged();
  vsfNodesSet(r.nodes);
  status("Deleted from your filters", "ok");
}
async function vsfNodeExport(f) {
  const name = (f.label || "filter").trim();
  const text = JSON.stringify({ name, doc: f.doc || "", code: (f.p.args && f.p.args.code) || "", eva_filter: 1 }, null, 1);
  const p = await window.api.saveText(name.replace(/[<>:"/\|?*]/g, "_") + ".json", text, [{ name: "EVA toolkit custom filter", extensions: ["json"] }]);
  if (p) status("Exported: " + p, "ok");
}

// a filter in a group, as a card (VapourKit-style): number, drag handle, name, open / close, on / off, duplicate,
// remove; open: its name, save to your filters / export / delete, what it does, then its code (or an app stage's settings)
const VK_ICON = {
  save: '<svg viewBox="0 0 16 16"><path d="M3 2h8l3 3v9H2V2zM5 2v4h6V2M5 14v-5h6v5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  export: '<svg viewBox="0 0 16 16"><path d="M8 2v8M4.5 6.5L8 10l3.5-3.5M2.5 12v2h11v-2" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  trash: '<svg viewBox="0 0 16 16"><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v5M9.2 6.5v5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  dup: '<svg viewBox="0 0 16 16"><rect x="5" y="5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M11 3.5V2H2v9h1.5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  lib: '<svg viewBox="0 0 16 16"><path d="M2 3h12l-4.5 5.5V13l-3 1.5v-6z" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
};
let VK_DRAG = null;                                 // {gid, i} of the card being dragged
function vsfFilterRow(g, i) {
  const f = g.filters[i], open = vsfOpen[g.id] === i, isCode = f.type === "code";
  const card = el("div", "vk-card" + (f.on === false ? " off" : "") + (open ? " open" : ""));
  const ib = (svg, tip, fn, cls = "") => { const b = el("button", "vk-ib " + cls); b.innerHTML = svg; b.title = tip; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
  const toggle = () => { vsfOpen[g.id] = open ? -1 : i; renderVsGroups(); };
  // ---- the head
  const head = el("div", "vk-head");
  const num = el("span", "vk-num", String(i + 1));
  const grip = el("span", "vk-grip", "⠿"); grip.title = "Drag to change the order"; grip.draggable = true;
  const startDrag = (e) => { VK_DRAG = { gid: g.id, i }; e.dataTransfer.setData("application/x-vsf-filter", String(i)); e.dataTransfer.setData("application/x-eva-vsf", `${g.id}|${i}`); e.dataTransfer.effectAllowed = "copyMove"; card.classList.add("dragging"); };
  const endDrag = () => { VK_DRAG = null; card.classList.remove("dragging"); for (const c of $$(".vk-card.drop-a, .vk-card.drop-b")) c.classList.remove("drop-a", "drop-b"); };
  grip.ondragstart = (e) => { e.stopPropagation(); startDrag(e); };
  head.draggable = true;                             // grab the card anywhere on its header: the dots are no longer the only handle
  head.addEventListener("dragstart", (e) => { if (e.target === head) startDrag(e); });
  head.addEventListener("dragend", endDrag);
  grip.title = "Drag to change the order - or onto the Timeline to run it only there (1/30 of the video long, from where you drop it)";
  grip.ondragend = () => { VK_DRAG = null; card.classList.remove("dragging"); for (const c of $$(".vk-card.drop-a, .vk-card.drop-b")) c.classList.remove("drop-a", "drop-b"); };
  card.addEventListener("dragover", (e) => {
    if (!VK_DRAG || VK_DRAG.gid !== g.id) return;
    e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = "move";
    const r = card.getBoundingClientRect(), below = e.clientY > r.top + r.height / 2;
    card.classList.toggle("drop-a", !below); card.classList.toggle("drop-b", below);
  });
  card.addEventListener("dragleave", () => card.classList.remove("drop-a", "drop-b"));
  card.addEventListener("drop", (e) => {
    if (!VK_DRAG || VK_DRAG.gid !== g.id) return;
    e.preventDefault(); e.stopPropagation();
    const below = card.classList.contains("drop-b"), from = VK_DRAG.i;
    card.classList.remove("drop-a", "drop-b"); VK_DRAG = null;
    let to = i + (below ? 1 : 0); if (from < to) to--;
    if (to === from) return;
    const [m] = g.filters.splice(from, 1); g.filters.splice(to, 0, m);
    vsfOpen[g.id] = vsfOpen[g.id] === from ? to : -1;
    vsgChanged();
  });
  const title = el("b", "vk-title", f.label || f.p.call); title.title = (f.from || f.p.call || "") + " · click to show / hide";
  title.onclick = toggle;
  const chev = el("button", "vk-ib vk-chev"); chev.innerHTML = `<svg viewBox="0 0 16 16"><path d="${open ? "M4 10l4-4 4 4" : "M4 6l4 4 4-4"}" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>`; chev.title = open ? "Close" : "Open"; chev.onclick = (e) => { e.stopPropagation(); toggle(); };
  const on = el("label", "switch vk-sw"), onCb = document.createElement("input");      // the same switch as the group's
  onCb.type = "checkbox"; onCb.checked = f.on !== false; on.title = "On / off";
  onCb.onchange = () => { f.on = onCb.checked; vsgChanged(); };
  on.append(onCb, el("span"));
  let engBtn = null;
  if (f.type === "ai") {                             // a model node: Engine sits in the header (its name is the title, its file is in the settings)
    engBtn = el("button", "vk-eng", "⚙ Engine"); engBtn.title = "TensorRT engine settings (precision, shapes, build command)";
    engBtn.onclick = (e) => { e.stopPropagation(); vsfEngineOpen(g, i, false); };
  }
  head.append(num, grip, title, el("span", "push"), ...(engBtn ? [engBtn] : []), chev, on,
    ib(VK_ICON.dup, "Duplicate", () => {
      if (g.filters.length === 1) {              // alone in its group: the copy is a group of its own, right after it
        const c = structuredClone(g); c.id = vsfNewGroup().id; if (c.filters[0]) delete c.filters[0].node;
        const at = S.vs_groups.findIndex((x) => x.id === g.id); S.vs_groups.splice(at + 1, 0, c); vsgChanged(); return;
      }
      g.filters.splice(i + 1, 0, Object.assign(structuredClone(f), { node: undefined })); vsfOpen[g.id] = i + 1; vsgChanged();
    }),
    ib("✕", "Remove this filter", () => {
      g.filters.splice(i, 1); vsfOpen[g.id] = -1;
      if (!g.filters.length) S.vs_groups = S.vs_groups.filter((x) => x.id !== g.id);   // an empty group is nothing
      vsgChanged();
    }, "vk-x"));
  head.ondblclick = (e) => { if (e.target === head) toggle(); };
  card.appendChild(head);
  if (!open) return card;
  // ---- open: name + its buttons
  const body = el("div", "vk-body");
  const row = el("div", "vk-namerow");
  const nm = el("input", "vk-name"); nm.value = f.label || ""; nm.placeholder = "Name"; nm.spellcheck = false;
  nm.onchange = () => { f.label = nm.value.trim() || f.label; vsgChanged(); };
  nm.onkeydown = (e) => { if (e.key === "Enter") nm.blur(); };
  row.appendChild(nm);
  if (isCode) {
    row.append(ib(VK_ICON.save, f.node ? `Save the changes to "${f.node}" in your filters` : "Save as your own filter (library › Your own filters)", () => { f.label = nm.value.trim(); vsfNodeSave(g, f); }, f.node ? "saved" : ""),
               ib(VK_ICON.export, "Export as a file (.json) - add it on another computer with Your own filters › + Add file", () => vsfNodeExport(f)));
    if (f.node) row.appendChild(ib(VK_ICON.trash, `Delete "${f.node}" from your filters in the library`, () => vsfNodeDelete(g, f), "vk-del"));
  }
  if (f.type !== "ai") body.appendChild(row);
  // ---- what it does (yours: write it, it is shown in the library)
  if (isCode) {
    const d = el("textarea", "vk-desc edit"); d.value = f.doc || ""; d.rows = 2; d.spellcheck = false;
    d.placeholder = "What it does (shown in the library when you save it)";
    d.onchange = () => { f.doc = d.value.trim(); vsgChanged(); };
    body.appendChild(d);
  } else if (f.doc && f.type !== "ai") body.appendChild(el("div", "vk-desc", f.doc));
  // ---- the code, or the settings
  if (isCode) {
    body.appendChild(codeEditor((f.p.args && f.p.args.code) || f.p.code || "clip = clip", (v) => { f.p.args = { ...(f.p.args || {}), code: v }; vsgChanged(); }, true));
    body.appendChild(el("div", "vk-foot", "`clip` in, `clip` out · Ctrl+Enter or clicking away applies it"));
  } else if (f.eva) body.appendChild(blockForm(f.type, f.p, () => vsgChanged()));
  else {
    const form = el("div", "vsf-args"), spec = f.argspec || [];
    if (!spec.length) form.appendChild(el("div", "vsf-doc", "No settings."));
    for (const a of spec) {
      const lab = el("label", null, a.name); lab.title = a.name;
      const inp = el("input"); inp.type = "text"; inp.spellcheck = false;
      inp.value = f.p.args && f.p.args[a.name] !== undefined ? f.p.args[a.name] : ""; inp.placeholder = a.hint || "";
      inp.onchange = () => { f.p.args = { ...(f.p.args || {}), [a.name]: inp.value }; vsgChanged(); };
      form.append(lab, inp);
    }
    if (f.type !== "ai") form.appendChild(el("div", "vsf-doc", f.type === "vs" ? "Empty = the plugin's default. Lists: comma separated (e.g. 1,0,0)." : "Empty = the default. Values are Python (e.g. 2.0, True, [1, 2], 'text')."));
    body.appendChild(form);
  }
  card.appendChild(body);
  return card;
}


// ---- TensorRT engine dialog (VapourKit-style) for AI model filters: the switches write a trtexec-style command, and
// the command is what is built (aiupscale.engine_settings / parse_args). Stored on the filter: f.p.engine.
const ENG = { gid: null, i: -1, info: {}, e: null, isNew: false, strength: 1 };   // the group by id: saving replaces S.vs_groups with copies
function engTarget() {
  const g = (S.vs_groups || []).find((x) => x.id === ENG.gid);
  return g && g.filters[ENG.i] ? { g, f: g.filters[ENG.i] } : {};
}
function engineArgs(e, i) {
  const nm = i.input_name || "input";
  const vsr = e.type === "vsr", T = e.frames || i.frames || 5;
  const dims = (h, w) => (vsr ? (i.layout === "4d" ? `1x${3 * T}x${h}x${w}` : `1x${T}x3x${h}x${w}`) : `1x3x${h}x${w}`);
  let s;
  if (i.fixed) s = `--shapes=${nm}:${dims(i.fixed[1], i.fixed[0])}`;                       // the model has one size
  else if (e.shape === "static") s = `--shapes=${nm}:${dims("H", "W")}`;                   // one engine per picture size
  else s = `--minShapes=${nm}:${dims(240, 240)} --optShapes=${nm}:${dims(720, 1280)} --maxShapes=${nm}:${dims(1080, 1920)}`;
  s += ` --saveEngine=OUTPUT_PATH --builderOptimizationLevel=${e.opt_level ?? 3}`;
  if (e.workspace && +e.workspace !== 4096) s += ` --memPoolSize=workspace:${e.workspace}M`;
  if (e.precision === "fp16") s += " --fp16"; else if (e.precision === "bf16") s += " --bf16";
  return s;
}
function parseEngineArgs(text) {
  const o = { precision: "fp32" }, unknown = [];
  let shapes = null;
  for (const tok of String(text).split(/\s+/).filter(Boolean)) {
    const [k, v = ""] = tok.split(/=(.*)/s);
    if (["--minShapes", "--optShapes", "--maxShapes", "--shapes"].includes(k)) { shapes = shapes || {}; shapes[k] = v.split(":").pop().split("x"); }
    else if (k === "--builderOptimizationLevel") o.opt_level = parseInt(v, 10);
    else if (k === "--fp16") o.precision = "fp16";
    else if (k === "--bf16") o.precision = "bf16";
    else if (k === "--memPoolSize") { const n = v.split(":").pop(); o.workspace = Math.round(parseFloat(n) * (/g/i.test(n) ? 1024 : 1)); }
    else if (!["--saveEngine", "--onnx", "--noTF32"].includes(k)) unknown.push(tok);
  }
  if (shapes) {
    o.shape = shapes["--shapes"] ? "static" : "dynamic";
    const d = shapes["--optShapes"] || shapes["--shapes"];
    if (d) { o.type = d.length === 5 || (d.length === 4 && +d[1] > 3) ? "vsr" : "image"; if (o.type === "vsr") o.frames = d.length === 5 ? +d[1] : (+d[1]) / 3; }
  }
  o.unknown = unknown;
  return o;
}
async function vsfEnginePending(path, label, settings, cb) {       // the same window, for a model in the library: Save gives the settings back
  let info = {};
  try { info = await rpc("model_info", { path }); } catch { /* shown below */ }
  if (info.error) { status(info.error, "err"); return; }
  Object.assign(ENG, { gid: null, i: -1, info, isNew: false, label, strength: 1, cb });
  const e = { type: info.vsr ? "vsr" : "image", shape: info.fixed ? "static" : "dynamic", precision: info.precision === "bf16" ? "bf16" : "fp16",
              opt_level: 3, workspace: 4096, frames: info.frames || 0, ...(settings || {}) };
  if (!e.args) e.args = engineArgs(e, info);
  ENG.e = e;
  $("engModel").textContent = label || path; $("engModel").title = path;
  const box = $("engInfo"); box.textContent = "";
  const add = (t) => box.appendChild(el("span", "badge", t));
  add(info.kind === "onnx" ? "ONNX" : `PyTorch · ${info.arch}`); if (info.scale) add(`${info.scale}x`);
  add(info.fixed ? `fixed input ${info.fixed[0]}x${info.fixed[1]}` : "any input size");
  if (info.vsr) add(`video model: ${info.frames || "?"} frames in`);
  add(`stored as ${info.precision}`);
  engineUi();
  $("engineDlg").classList.remove("hidden");
}
async function vsfEngineOpen(g, i, isNew) {
  g = (S.vs_groups || []).find((x) => x.id === g.id) || g;
  const f = g.filters[i];
  let info = {};
  try { info = await rpc("model_info", { path: f.p.args.model }); } catch { /* shown below */ }
  if (info.error) { status(info.error, "err"); if (isNew) { g.filters.splice(i, 1); vsgChanged(); } return; }
  Object.assign(ENG, { gid: g.id, i, info, isNew, label: f.label, strength: parseFloat(f.p.args.strength ?? "1") || 1, cb: null });
  const e = { type: info.vsr ? "vsr" : "image", shape: info.fixed ? "static" : "dynamic", precision: info.precision === "bf16" ? "bf16" : "fp16",
              opt_level: 3, workspace: 4096, frames: info.frames || 0, ...(f.p.engine || {}) };
  if (!e.args) e.args = engineArgs(e, info);
  ENG.e = e;
  $("engModel").textContent = f.label || f.p.args.model; $("engModel").title = f.p.args.model;
  const box = $("engInfo"); box.textContent = "";
  const add = (t) => box.appendChild(el("span", "badge", t));
  add(info.kind === "onnx" ? "ONNX" : `PyTorch · ${info.arch}`); if (info.scale) add(`${info.scale}x`);
  add(info.fixed ? `fixed input ${info.fixed[0]}x${info.fixed[1]}` : "any input size");
  if (info.vsr) add(`video model: ${info.frames || "?"} frames in`);
  add(`stored as ${info.precision}`);
  engineUi();
  $("engineDlg").classList.remove("hidden");
}
function engineUi(fillText = true) {
  const e = ENG.e, i = ENG.info || {};
  const seg = (id, v) => { for (const b of $(id).children) b.classList.toggle("on", b.dataset.v === String(v)); };
  seg("engType", e.type); seg("engShape", e.shape); seg("engPrecision", e.precision);
  $("engShape").querySelector('[data-v="dynamic"]').disabled = !!i.fixed;
  $("engType").querySelector('[data-v="vsr"]').disabled = i.kind !== "onnx";
  $("engTypeTxt").textContent = e.type === "vsr" ? `VSR (${e.frames || i.frames || 5} frames in, the middle one out)` : "Image (single frame)";
  $("engShapeTxt").textContent = i.fixed ? `Static (the model has one size: ${i.fixed[0]}x${i.fixed[1]})` : e.shape === "dynamic" ? "Dynamic (multiple resolutions)" : "Static (one engine per picture size)";
  $("engPrecTxt").textContent = { fp16: "FP16 (build + inference, recommended)", bf16: "BF16 (fp32's number range: no overflow)", fp32: "FP32 (exact, slower)" }[e.precision] || "";
  if (fillText && document.activeElement !== $("engArgs")) $("engArgs").value = e.args || "";

  const p = parseEngineArgs($("engArgs").value);
  $("engArgsNote").textContent = p.unknown.length ? `Not understood (ignored): ${p.unknown.join(" ")}` : "";
  $("engArgsNote").classList.toggle("err", !!p.unknown.length);
  $("engNote").textContent = (e.precision !== "fp32" ? `The first frame checks the ${e.precision} engine against fp32; if the picture differs by more than 0.75 levels on average, fp32 is used for this model. ` : "")
    + (e.shape === "dynamic" && !i.fixed ? "Dynamic: one engine for every picture size in the range (no rebuild when the size changes); if TensorRT cannot build the whole range, a static engine per size is used. " : "")
    + (i.kind !== "onnx" ? "PyTorch models (.pth / .safetensors): the engine is built per picture size; the precision applies. " : "")
    + "Engines are kept in engines\\.";
}
function engineDone(save, build) {
  $("engineDlg").classList.add("hidden");
  if (ENG.cb) {                                        // opened from the model library: hand the settings back
    const cb = ENG.cb; ENG.cb = null;
    if (save) cb({ ...ENG.e }, !!build);
    return;
  }
  const { g, f } = engTarget(), i = ENG.i;
  if (!f) return;
  if (!save) { if (ENG.isNew) { g.filters.splice(i, 1); vsfOpen[g.id] = -1; vsgChanged(); } return; }
  f.p.engine = { ...ENG.e };
  f.p.args = { ...f.p.args, strength: String(ENG.strength) };
  delete f.p.args.preset;                          // the engine settings replace the preset
  vsgChanged();
  if (build) {
    const w = (S.info && S.info.width) || 720, h = (S.info && S.info.height) || 480;
    status(`Building the TensorRT engine for ${f.label} (${w}x${h}) - see the log`);
    rpc("build_ai_engine", { model: f.p.args.model, engine: f.p.engine, w, h, backend: S.backend })
      .then((r) => { if (r && r.error) { status(r.error, "err"); log(r.error, "err"); } else status(`Engine ready: ${f.label}`, "ok"); })
      .catch(() => {});
  }
}
function initEngineDialog() {
  const pick = (id, key, num) => { for (const b of $(id).children) b.onclick = () => {
    if (b.disabled) return;
    ENG.e[key] = num ? +b.dataset.v : b.dataset.v;
    if (key === "type" && ENG.e.type === "vsr" && !ENG.e.frames) ENG.e.frames = ENG.info.frames || 5;
    ENG.e.args = engineArgs(ENG.e, ENG.info); engineUi(); }; };
  pick("engType", "type"); pick("engShape", "shape"); pick("engPrecision", "precision");
  $("engArgs").oninput = () => {                   // a hand-edited command: the switches follow it
    const p = parseEngineArgs($("engArgs").value);
    for (const k of ["precision", "opt_level", "workspace", "shape", "type", "frames"]) if (p[k] !== undefined && !Number.isNaN(p[k])) ENG.e[k] = p[k];
    ENG.e.args = $("engArgs").value; engineUi(false);
  };
  $("btnEngineClose").onclick = $("btnEngineCancel").onclick = () => engineDone(false);
  $("btnEngineSave").onclick = () => engineDone(true, false);
  $("btnEngineBuild").onclick = () => engineDone(true, true);
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("engineDlg").classList.contains("hidden")) engineDone(false); });
}

// the filter browser (+ Add filter): search, categories, a table of every filter with its settings and what it does,
// details of the selected one, Add. Data: the EVA toolbox library (names, groups with descriptions, arguments, formats).
const VSFB = { g: null, cat: "all", sel: null };
function vsfAddTo(g, f, onAdd) {
  if (onAdd) { onAdd(f); return; }                 // Chain mode: a block, not a group filter
  if (f.kind === "code") {                         // + Custom / new filter: a blank filter of your own, to write and save
    g.filters.push({ type: "code", on: true, label: "New filter", doc: "", argspec: [{ name: "code", hint: "Python: change `clip`" }],
                     p: { call: "code", args: { code: VSF_NEW_CODE } } });
    vsfOpen[g.id] = g.filters.length - 1;
    vsgChanged();
    return;
  }
  if (f.kind === "vs" || f.kind === "jet" || f.kind === "my") {     // it comes in as code: its settings as variables with defaults
    g.filters.push({ type: "code", on: true, label: f.name || f.call, from: f.call, doc: (f.doc || f.note || "").slice(0, 300),
                     argspec: [{ name: "code", hint: "Python: change `clip`" }], p: { call: "code", args: { code: filterTemplate(f) } } });
    vsfOpen[g.id] = g.filters.length - 1;
    vsgChanged();
    return;
  }
  const p = { call: f.call, args: { ...(f.defaults || {}) } };
  for (const k of ["formats", "adapt", "prep"]) if (f[k] !== undefined) p[k] = structuredClone(f[k]);
  g.filters.push({ type: f.kind, on: true, label: f.name || f.call, doc: (f.doc || f.note || vsfCatInfo(f.group)[1] || "").slice(0, 300), argspec: vsfArgsOf(f), p });
  vsfOpen[g.id] = g.filters.length - 1;
  const chosen = f.kind === "ai" && PICK.eng && PICK.eng[f.path];   // the engine chosen in the model library (Automatic / a built one / Edit…): nothing more to ask
  if (chosen) g.filters[g.filters.length - 1].p.engine = structuredClone(chosen);
  vsgChanged();
}
function vsfAddBlock(g, f) {                       // the library's EVA stages and code templates as a group entry
  if (f.kind === "node") g.filters.push({ type: "code", on: true, label: f.name, node: f.name, doc: f.doc || "", argspec: [{ name: "code", hint: "Python: change `clip`" }],
                                           p: { call: "code", args: { code: f.code } } });
  else if (f.kind === "tpl") g.filters.push({ type: "code", on: true, label: f.name.split(" (")[0], doc: f.doc, argspec: [{ name: "code", hint: "Python: change `clip`" }],
                                          p: { call: "code", args: { code: f.code } } });
  else g.filters.push({ type: f.id, eva: true, on: true, label: f.name, doc: (f.doc || "").slice(0, 300), p: {} });
  vsfOpen[g.id] = g.filters.length - 1;
  vsgChanged();
}
function vsfRequired(f) {
  return (f.args || []).filter((a) => (f.kind === "vs" ? !a.opt && !a.auto : a.required) && !(f.defaults && f.defaults[a.name] !== undefined)).map((a) => a.name);
}
function vsfCatInfo(key) {                        // [name, description] of a category
  const c = (VSF_GROUPS || []).find((x) => x[0] === key);
  return c ? [c[1], c[2] || ""] : [key || "Other", ""];
}
function vsfMatches(f, words) {
  const [cn, cd] = vsfCatInfo(f.group);
  const t = `${f.name} ${f.call} ${f.plugin || ""} ${cn} ${cd} ${f.doc || ""} ${f.note || ""} ${(f.args || []).map((a) => a.name).join(" ")}`.toLowerCase();
  return words.every((w) => t.includes(w));
}

function vsfGroupEl(g, gi) {
  if (g.filters.length === 1 && g.on === false) { g.filters[0].on = false; g.on = true; }   // one filter: its own switch is the only one
  const box = el("div", "group vsf-group" + (g.on === false ? " off" : ""));
  box.dataset.gid = g.id;
  const head = el("div", "group-head");
  const grip = el("span", "vsf-grip", "⠿"); grip.title = "Drag to where this group should run";
  grip.draggable = true;
  grip.ondragstart = (e) => { e.dataTransfer.setData("text/plain", g.id); e.dataTransfer.setData("application/x-eva-vsf", `${g.id}|`); e.dataTransfer.effectAllowed = "copyMove"; box.classList.add("dragging"); };
  grip.ondragend = () => { box.classList.remove("dragging"); vsfDropMark(null); };
  const title = el("h3", null, "VapourSynth filters");
  const where = el("div", "vsf-where", vsfPoint(g) === "ivtc" ? "Runs instead of the app's IVTC" : "Runs " + vsfWhere(g));
  const sw = el("label", "switch"); const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = g.on !== false;
  cb.onchange = () => { g.on = cb.checked; vsgChanged(); }; sw.append(cb, el("span")); sw.title = "This group on / off";
  const btn = (t, tip, fn) => { const b = el("button", "ghost vsf-gbtn", t); b.title = tip; b.onclick = fn; return b; };
  const add = btn("+ Add filter", "Browse and search every installed VapourSynth plugin and vs-jetpack function",
    () => vsfBrowser(g));
  const dup = btn("⧉", "Duplicate this group (with its filters)", () => {
    const c = structuredClone(g); c.id = vsfNewGroup().id;
    S.vs_groups.splice(gi + 1, 0, c); vsgChanged();
  });
  const del = btn("✕", "Delete this group", () => { S.vs_groups.splice(gi, 1); vsgChanged(); });
  const tools = el("span", "vsf-gtools"); tools.append(sw, dup, del);   // adding is the top "+ Add a filter" (a new group runs after the last one here)
  head.append(grip, title, tools);
  box.append(head, where);
  if (vsfIvtcSpot(g)) {                          // next to Telecine: this group can do the IVTC itself
    const f = el("div", "field");
    const lab = el("label", null, "IVTC"); lab.title = "Replace the app's IVTC / PAL handling with this group";
    const s2 = el("label", "switch"); const c2 = document.createElement("input"); c2.type = "checkbox"; c2.checked = !!g.own_ivtc;
    c2.onchange = () => { g.own_ivtc = c2.checked; vsgChanged(); };
    s2.append(c2, el("span"), el("em", null, "do the IVTC myself (replaces the app's)"));
    f.append(lab, s2); box.appendChild(f);
    if (g.own_ivtc) box.appendChild(el("div", "note", "This group gets the raw interlaced picture and must give progressive frames, e.g. vivtc.VFM (order 1) + vivtc.VDecimate, or tivtc.TFM + tivtc.TDecimate, or bwdif.Bwdif. The export uses its frame rate (constant), not the app's VFR."));
  }
  const list = el("div", "vsf-list");
  g.filters.forEach((_, i) => list.appendChild(vsfFilterRow(g, i)));
  if (g.filters.length === 1) {                  // one filter: its card is the whole thing (no header above it saying the same)
    box.classList.add("vsf-solo");
    const cg = list.querySelector(".vk-grip");
    if (cg) {                                    // its handle moves the group to where it should run
      cg.title = (vsfPoint(g) === "ivtc" ? "Runs instead of the app's IVTC" : "Runs " + vsfWhere(g)) + " - drag it up or down to run it somewhere else, or onto the Timeline to run it only there (1/30 of the video long, from where you drop it)";
      cg.ondragstart = (e) => { e.stopPropagation(); e.dataTransfer.setData("text/plain", g.id); e.dataTransfer.setData("application/x-eva-vsf", `${g.id}|0`); e.dataTransfer.effectAllowed = "copyMove"; box.classList.add("dragging"); };
      cg.ondragend = () => { box.classList.remove("dragging"); vsfDropMark(null); };
    }
    const vh = list.querySelector(".vk-head");
    if (vh) {
      vh.addEventListener("dragstart", (e) => { if (e.target !== vh) return; e.dataTransfer.setData("text/plain", g.id); e.dataTransfer.effectAllowed = "copyMove"; box.classList.add("dragging"); });
      vh.addEventListener("dragend", () => { box.classList.remove("dragging"); vsfDropMark(null); });
    }
  }
  if (!g.filters.length) list.appendChild(el("div", "note", "Empty. Add something with + Add a filter at the top."));
  box.appendChild(list);
  return box;
}

function vsfDropMark(beforeEl) {                  // the line that shows where a dragged group will land
  let m = document.getElementById("vsfDropMark");
  if (!beforeEl) { if (m) m.remove(); return; }
  if (!m) { m = el("div", "vsf-drop"); m.id = "vsfDropMark"; }
  beforeEl.parentNode.insertBefore(m, beforeEl);
}
function vsfSlotAt(y, st = vsfStaticGroups()) {   // mouse y -> before which Repair group (index into st)
  for (let k = 0; k < st.length; k++) { const r = st[k].getBoundingClientRect(); if (y < r.top + r.height / 2) return k; }
  return st.length;
}
function vsfBindDrop() {
  const panel = document.querySelector('[data-panel="adjust"]');
  if (!panel || panel.dataset.vsfDrop) return;
  panel.dataset.vsfDrop = "1";
  panel.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes("text/plain")) return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    const st = vsfStaticGroups().filter((x) => !x.classList.contains("dragging")), slot = vsfSlotAt(e.clientY, st);
    vsfDropMark(st[slot] || document.querySelector('[data-panel="adjust"] .vsf-new'));
  });
  panel.addEventListener("dragleave", (e) => { if (!panel.contains(e.relatedTarget)) vsfDropMark(null); });
  panel.addEventListener("drop", (e) => {
    const id = e.dataTransfer.getData("text/plain");
    const st = vsfStaticGroups().filter((x) => !x.classList.contains("dragging"));
    const slot = vsfSlotAt(e.clientY, st), after = slot <= 0 ? "" : st[slot - 1].dataset.gid;
    vsfDropMark(null);
    if (id.startsWith("rep:")) {                  // a Repair group itself
      e.preventDefault();
      const step = id.slice(4);
      if (after === "" || after === "telecine") { status("Telecine & interlacing and Composite artifacts always run first."); return; }
      const o = repairOrder().filter((x) => x !== step);
      o.splice(after === "composite" ? 0 : o.indexOf(after) + 1, 0, step);
      if (o.join() !== repairOrder().join()) {
        changed("repair_order", o);
        const k = o.indexOf(step);
        log(`${STEP_NAME[step].replace(/^the /, "")} now runs ${k ? "after " + STEP_NAME[o[k - 1]] : "right after Composite artifacts"}`, "ok");
        const sz = o.indexOf("size"), later = o.slice(sz + 1).filter((x) => ["compression", "halos", "colour"].includes(x));
        if (later.length) status(`${later.map((x) => STEP_NAME[x]).join(", ")} now work${later.length > 1 ? "" : "s"} at the output size: slower, and tuned for DVD size.`);
      }
      return;
    }
    const k = S.vs_groups.findIndex((g) => g.id === id);
    if (k < 0) return;
    e.preventDefault();
    const [g] = S.vs_groups.splice(k, 1);
    g.after = after; delete g.slot;
    if (!vsfIvtcSpot(g)) g.own_ivtc = false;
    S.vs_groups.push(g);                          // last of the groups at that place
    vsgChanged();
  });
}

function renderVsGroups() {
  const panel = document.querySelector('[data-panel="adjust"]');
  if (!panel) return;
  vsfMigrate();
  vsfBindDrop();
  for (const old of panel.querySelectorAll(".vsf-group")) old.remove();
  // the movable Repair groups in their order, after Composite artifacts
  let prev = panel.querySelector('[data-gid="composite"]');
  for (const step of repairOrder()) {
    const g = panel.querySelector(`[data-gid="${step}"]`);
    if (!g || !prev) continue;
    if (prev.nextElementSibling !== g) prev.after(g);
    prev = g;
  }
  const st = vsfStaticGroups(), tail = panel.querySelector(".vsf-new");
  S.vs_groups.forEach((g, gi) => {
    const a = vsfAfter(g), i = a === "" ? 0 : st.findIndex((x) => x.dataset.gid === a) + 1;
    const anchor = st[i] || tail;
    anchor.parentNode.insertBefore(vsfGroupEl(g, gi), anchor);
  });
}

function isProgressive() { return !!S.info && (S.info.image || (S.scan ? S.scan === "progressive" : !!S.info.progressive)); }
function sync() {
  setTimeout(() => { stepSummaries(); renderRangeBoxes(); rmodeSync(); }, 0);
  const pal = !!(S.info && S.info.pal);
  document.body.classList.toggle("scan-prog", isProgressive());
  document.body.classList.toggle("mode-bob", S.mode === "bob" && !pal && !isProgressive());   // 60p: no field matching
  if (S.info && !S.info.image) {
    const sc = S.info.scan || {}, det = S.info.progressive ? "progressive" : "interlaced";
    $("scanNote").textContent = isProgressive()
      ? "Whole frames: no field matching, deinterlacing or VFR analysis - " + (S.info.vfr_hint ? "every frame keeps its own time (this file mixes frame rates)." : "the frame rate is kept.") +
        (det !== "progressive" ? " (Detected as interlaced - you set Progressive.)" : sc.by === "picture" ? ` (Detected from the picture: ${(sc.combed * 100).toFixed(1)} % of ${sc.frames} frames combed.)` : " (Detected from the file's flags.)")
      : "";
  }
  for (const e of $$(".ntsc")) e.classList.toggle("hidden", pal);
  for (const e of $$(".pal")) e.classList.toggle("hidden", !pal);
  for (const c of $$("[data-bind]")) {
    const v = getv(c.dataset.bind);
    if (c.classList.contains("seg")) for (const b of c.children) b.classList.toggle("on", b.dataset.v === String(v));
    else if (c.classList.contains("stepper")) { const i = c.querySelector("input"); if (document.activeElement !== i) i.value = v; }
    else if (c.type === "checkbox") c.checked = !!v;
    else if (document.activeElement !== c) c.value = v;
  }
  for (const f of $$("[data-show]"))          // a rule never shows an NTSC-only field on PAL (or the reverse)
    f.classList.toggle("hidden", !showRule(f.dataset.show) || (pal && f.classList.contains("ntsc")) || (!pal && f.classList.contains("pal")));
  for (const f of $$(".field[data-key]")) {
    const k = f.dataset.key, chip = f.querySelector(".chip"), has = k in S.auto;
    const cur = k.startsWith("grade.") ? S.gradeAll[k.slice(6)] : getv(k);
    const same = has && String(S.auto[k]) === String(cur);
    chip.classList.toggle("hidden", !has);
    chip.className = "chip " + (!has ? "hidden" : same ? "auto" : "edited");
    chip.textContent = same ? "auto" : "edited ↺";
    chip.title = has && !same ? `Back to the analysed value (${S.auto[k]})` : "";
    f.classList.toggle("edited", has && !same);
  }
  repairNetLabels();
  renderVsGroups();
  $("aiNote").textContent = {
    field: "FieldNet: one network for the whole cleanup - its own IVTC repair from the raw fields, dot crawl, rainbows, MPEG blocking, halos and colour bleed, recurrent over time. After it the picture is steadied (motion-compensated averaging, fine-detail and still-area smoothing, and a long still-area lock), so shimmer, flicker and wobble on still detail such as door mesh or rug lettering go away without losing sharpness. Each dial: 1 = remove fully, 0 = leave as on the disc; while a dial is above 0 the matching classic step below is left out. Grain keep 1 = film grain and noise stay. Detail recovery 1 = fine detail an old decoder cut out is rebuilt.",
    uni: "All-in-one network: dot crawl, rainbows, MPEG blocking / flicker, halos and colour bleed / shift in one pass, recurrent over time, so it flickers less than the other cleaners - for fully steady still detail (door mesh, rug lettering) add the Steady chain stabilizer. Each dial: 1 = remove fully, 0 = leave as on the disc. While a dial is above 0 the matching classic step (MPEG repair, halo removal, colour shift / bleed fix) is left out. Grain keep 1 = film grain and noise stay. Compared with StableNet it is a little softer on the finest detail.",
    stable: "StableNet: recurrent network for dots, rainbows and MPEG flicker; steady by itself, no stabilizer needed. It also removes film grain.",
    temporal: "Neural network that looks at 5 frames: dots and rainbows invert from frame to frame, real detail does not. With it on, the classic luma filters are left out (they cost detail); the classic rainbow filter stays.",
    xnet: "XNet: the 5-frame network trained for steadiness (no wobble on fine texture) and to leave grain and detail alone; its luma correction can only act where dot crawl can exist.",
    precise: "Single-frame version of the same network (use if the temporal one shows ghosting on a source).",
    off: "Only the classic temporal filters (they soften fine detail - the AI is better).",
  }[S.ai_model] || "";
  // the dots + rainbows models take their strength from Dot crawl / Rainbows: with both Off they do not run at all
  // (vsgraph._ai_cleaner) - the Simpsons S1 preview then showed the disc's own crawl / rainbow flicker unfiltered
  if (["precise", "temporal", "xnet", "stable"].includes(S.ai_model) && S.crawl === "off" && S.rainbow === "off")
    $("aiNote").textContent += " Dot crawl and Rainbows are both Off, so this cleanup does not run now - press Auto (Composite artifacts) or set them.";
  $("frameNote").textContent = [
    { off: "", repair: "Anti-aliasing runs only where a frame had to be rebuilt from one field.", all: "Anti-aliasing runs on every frame: use it for sources that were deinterlaced badly before they got to you." }[S.aa],
    { off: "", ai: "AI blend removal: every blended frame is rebuilt as the picture of its own moment from the frames around it - no repeated frames, so pans stay smooth. Clean frames, fades and dissolves are left as they are.",
      dup: "Blended frames are replaced by the nearer clean frame: a repeated frame, which can make pans stutter.", interp: "Blended frames are replaced by a motion-interpolated frame: smoother, can smear on fast motion." }[S.deblend],
  ].filter(Boolean).join(" ");
  $("stabNote").textContent = {
    off: "Removes line wobble, shimmer and flicker that remain after cleaning.",
    chain: "The steadiest: motion-compensated averaging, then fine-detail averaging, then still-area smoothing, then a long still-area lock that holds still detail (rug lettering, door mesh) completely steady. Real Courage disc after the AI cleanup: door and rug flicker 1.1 -> 0.1, lettering shimmer 0.65 -> 0.03, wobble 1.1 -> 0.34, sharpness and colour unchanged. Runs at 16 bits. Light / Normal / Strong set how hard each part averages.",
    ustab: "Steadies flicker, leftover crawl, line wobble and noise in every kind of shot. Fine detail is averaged over 4 motion-compensated frames each side, brightness only where it changes less than flicker does, and still areas over 10 frames each side in balanced pairs, so fades, cross-fades, flashes, light changes and cuts are never dragged into the next frame.",
    gtfix: "Steadies line wobble and shimmer from frame-by-frame filters. Only fine detail is averaged, and only where the motion-compensated neighbours match, so the picture is not blended or ghosted. CPU.",
    tfix_ai: "Made for line wobble / shimmer. Runs on TensorRT (GPU, ~64 fps at DVD size; first use per size builds an engine, ~1 min). Weaker on DVD flicker than TTempSmooth.",
    tfix_classic: "CPU version of TemporalFix, made for 2D animation. Slow but works everywhere.",
    mc: "Averages motion-compensated neighbouring frames: steadies grain and flicker, keeps motion.",
    ttemp: "Smooths leftover dot crawl and flicker over time. Only useful without the AI cleanup: on top of it, it measured -3.5 dB and removed a third of the finest detail and grain.",
  }[S.stab] || "";
  setSegView();
  $("selZoom").value = S.zoom;
  // derived info
  if (S.info) {
    const c = S.crop, cw = S.info.width - c.l - c.r, ch = S.info.height - c.t - c.b;
    $("cropSize").textContent = `picture ${cw}x${ch}`;
    const darC = srcDar() * (cw / S.info.width) / (ch / S.info.height);
    const size = S.opts.sizes.find((s) => s.key === S.size) || S.opts.sizes[0];
    const even = (v) => Math.round(v / 2) * 2;
    let note = "";
    if (size.mode === "sq") $("outRes").textContent = `→ ${even(size.h * darC)}x${size.h}, square pixels`;
    else if (size.mode === "custom") {
      const h = even(S.custom_h), w = S.custom_lock ? even(h * darC) : even(S.custom_w);
      $("outRes").textContent = `→ ${w}x${h}, square pixels`;
      const off = (w / h) / darC;
      if (!S.custom_lock && Math.abs(off - 1) > 0.01)
        note = `${w}x${h} is ${(w / h).toFixed(3)}:1, the picture is ${darC.toFixed(3)}:1 - it will look ${off > 1 ? "wider" : "narrower"} by ${Math.abs(100 * (off - 1)).toFixed(1)} %.`;
    } else $("outRes").textContent = `→ ${cw}x${ch}, shown as ${even(ch * darC)}x${ch}`;
    if (S.aspect === "custom" && !parseRatio(S.aspect_custom)) note = "Enter the ratio as 1.85 or 16:9 (0.5 to 4). Until then the ratio from the disc is used.";
    $("sizeNote").textContent = note;
  }
  const L = S.levels;
  $("levelNote").textContent = L ? `Measured: darkest picture tone ${L.level_black}, brightest ${L.level_white}` + (L.level_whites != null ? `, painted whites ${L.level_whites}` : "") + ` (DVD black = 16, white = 235).` : "Press Auto colour to measure the black and white level of this source.";
  $("gamutNote").textContent = (L ? `Measured: ${L.gamut_out} % of pixels are stored stronger than a screen can show, ${L.clip_flat} % sit flat at the limit (worst tenth of frames). ` : "") + ({
    off: "", compress: "Compress: uses the colour the disc really stores beyond the range. Right for DVDs.",
    ai: "AI rebuild: for colour that was cut off before the disc or capture was made. It estimates the lost gradation from the other colour channels and the surroundings; where all three channels are flat there is nothing to go on.",
  }[S.grade.gamut] || "");
  $("wbNote").textContent = "";
  const trt = S.backend === "tensorrt";
  $("backendBadge").textContent = trt ? "TensorRT" : "PyTorch"; $("backendBadge").classList.toggle("trt", trt);
  $("backendNote").textContent = trt ? "Each model is compiled for this GPU the first time it meets a frame size (5-25 s each, then cached). The status line says when it happens."
                                     : "Models run directly in PyTorch.";
  $("exportNote").textContent = S.script ? "Export uses your edited script (Script step) for every ticked episode." : "";
  $("scriptFlag").classList.toggle("hidden", !S.script);
  $("srcShort").textContent = S.info ? basename(S.info.path) : "no source";
  $("sharpNote").textContent = (S.sharp.luma > 0 || S.sharp.chroma > 0)
    ? (S.sharp.mode === "faithful" ? "Faithful: edges get steeper but never overshoot the pixels around them, so no halos." : "Unsharp: crisper, but can add bright / dark rims along outlines - check Split.")
    : (sharpAfterSize() ? "Applied after resizing, so it acts on the final picture." : "Runs before Shape and size (moved in the Filtering order): it acts on the picture before resizing.");
  $("sharpWhere").textContent = sharpAfterSize() ? "at the output size" : "before resizing";
  encoderUi(); vfrButton(); syncColourUi();
  $("kernelNote").textContent = S.size === "keep" ? "" : ({
    mks2021: "Magic Kernel Sharp 2021: very little ringing or aliasing, crisp - a good default for upscaling cartoons.",
    mks2013: "Magic Kernel Sharp 2013: like 2021 with a shorter kernel, slightly softer.",
    bicubic: "b = blur, c = sharpness. Mitchell 0.33 / 0.33, Catmull-Rom 0 / 0.5, B-spline 1 / 0, Hermite 0 / 0.",
    ewa_lanczossharp: "Elliptical Lanczos (mpv's best upscaler): smooth diagonals, little aliasing. Runs on the GPU.",
    ewa_lanczos: "Elliptical Lanczos: smooth diagonals. Runs on the GPU.",
    ewa_robidouxsharp: "Elliptical Robidoux: softer than EWA Lanczos, almost no ringing. Runs on the GPU.",
  }[S.kernel] || "");
  renderQueue(); updateButtons();
}

function parseRatio(text) {
  const t = String(text).trim().replace(",", ".").replace("/", ":").replace("x", ":");
  const p = t.split(":");
  const v = p.length === 2 ? parseFloat(p[0]) / parseFloat(p[1]) : parseFloat(t);
  return v >= 0.5 && v <= 4 ? v : null;
}
// display aspect ratio of the whole source frame: from the disc or the chosen one (same rule as server.py)
function srcDar() {
  const v = S.aspect === "custom" ? parseRatio(S.aspect_custom) : S.aspect !== "auto" ? parseRatio(S.aspect) : null;
  return v || S.info.dar;
}

function updateButtons() {
  const has = !!S.info, busy = S.busy;
  document.body.classList.toggle("busy", busy);
  $("btnStop").classList.toggle("hidden", !busy);
  for (const b of $$(".auto-btn")) b.disabled = !S.info || S.busy;
  for (const id of ["btnAnalyze", "btnAutoColour", "btnAutoWB", "btnAutoCrop", "btnPlay", "btnTrimIn", "btnTrimOut", "btnReport", "btnVsview", "btnChapters", "btnWhole", "btnAddEp", "btnDelEp", "btnPrev", "btnNext"])
    $(id).disabled = !has || busy;
  for (const id of ["btnOpenFile", "btnOpenFolder", "btnChange", "btnImport"]) $(id).disabled = busy;
  $("btnExport").disabled = !has || busy || S.checked.size === 0;
  $("btnExport").disabled = !has || S.checked.size === 0;          // adding to the queue works while it runs
  $("btnExport").textContent = S.info && S.info.image ? "Add the image to the queue"
    : S.checked.size ? `Add ${S.checked.size} episode${S.checked.size === 1 ? "" : "s"} to the queue` : "Add to the queue";
  $("btnShowOut").disabled = !S.out_dir;
  for (const b of $$("#pageTabs button")) { b.classList.toggle("on", b.dataset.page === S.step); b.disabled = b.dataset.page !== "adjust" && !has; }
  for (const b of $$("#stepNav button")) {
    const st = b.dataset.step;
    b.classList.toggle("on", st === S.step);
    b.classList.toggle("done", st !== S.step && ((st === "source" && has) || (st === "analyze" && S.analyzed)));
    b.disabled = st !== "source" && !has;
  }
}

// ---- processing order of Colour / Crop & size (drag in the rail)
const STAGE_STEP = { colour: "colour", size: "frame" };
const STAGE_NAME = { colour: "Colour", size: "Crop & size" };
function stageOrder() {
  const o = (S.stage_order || []).filter((s) => s in STAGE_STEP);          // older settings may still list "upscale"
  return o.length === 2 && new Set(o).size === 2 ? o : ["colour", "size"];
}
function applyStageOrder() {
  const nav = $("stepNav"), anchor = nav.querySelector('[data-step="script"]');
  for (const s of stageOrder()) { const b = nav.querySelector(`[data-step="${STAGE_STEP[s]}"]`); if (b) nav.insertBefore(b, anchor); }   // Crop & size is part of Repair now
  [...nav.children].forEach((b, i) => { b.querySelector(".dot").textContent = String(i + 1); });
  // the ← / Next buttons under each panel follow the rail
  const steps = [...nav.children].map((b) => [b.dataset.step, b.querySelector(".lbl").firstChild.textContent.trim()]);
  steps.forEach(([st], i) => {
    const foot = document.querySelector(`.panel[data-panel="${st}"] .panel-foot`);
    if (!foot) return;
    const [back, next] = [foot.querySelector("button:not(.primary)[data-goto]"), foot.querySelector("button.primary[data-goto]")];
    if (back && i > 0) { back.dataset.goto = steps[i - 1][0]; back.textContent = `← ${steps[i - 1][1]}`; }
    if (next && i + 1 < steps.length) { next.dataset.goto = steps[i + 1][0]; next.textContent = `Next: ${steps[i + 1][1]} →`; }
  });
}
function initStageDrag() {
  const nav = $("stepNav");
  let drag = null;
  const stageOf = (b) => Object.keys(STAGE_STEP).find((k) => STAGE_STEP[k] === b.dataset.step);
  const clear = () => { for (const b of $$("#stepNav button")) b.classList.remove("drop-before", "drop-after", "dragging"); };
  for (const b of $$("#stepNav button.movable")) {
    b.addEventListener("dragstart", (e) => { drag = stageOf(b); e.dataTransfer.setData("text/plain", drag); e.dataTransfer.effectAllowed = "move"; b.classList.add("dragging"); });
    b.addEventListener("dragend", () => { drag = null; clear(); });
    b.addEventListener("dragover", (e) => {
      if (!drag) return;
      e.preventDefault();
      const r = b.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2;
      clear(); b.classList.add(after ? "drop-after" : "drop-before");
    });
    b.addEventListener("drop", (e) => {
      if (!drag) return;
      e.preventDefault();
      const r = b.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2, target = stageOf(b);
      const o = stageOrder().filter((s) => s !== drag);
      if (target !== drag) o.splice(o.indexOf(target) + (after ? 1 : 0), 0, drag); else o.splice(stageOrder().indexOf(drag), 0, drag);
      drag = null; clear();
      if (o.join() === stageOrder().join()) return;
      changed("stage_order", o); applyStageOrder();
      log("Processing order: " + o.map((s) => STAGE_NAME[s]).join(" → "), "ok");
    });
  }
  nav.addEventListener("dragover", (e) => { if (drag) e.preventDefault(); });
}

function goto(step) {
  // the workspace has three pages (Repair, Colour, Deliver); Source is the Media panel and Script a dock tab
  if (step === "frame" || step === "analyze") step = "adjust";
  if (step === "source") { showLeft("media"); if (S.step && S.step !== "source") { updateButtons(); return; } step = "adjust"; }
  if (step === "script") { showDock("script"); return; }
  if (step !== "adjust" && !S.info) return;
  S.step = step;
  document.body.classList.toggle("step-colour", step === "colour");   // scopes beside the picture
  document.body.classList.toggle("step-export", step === "export");
  if (step !== "colour") fitStageRow();
  if (step === "colour") requestScopes();
  setTimeout(layoutImage, 0);
  for (const p of $$(".insp .panel")) p.classList.toggle("hidden", p.dataset.panel !== step);
  $("inspTitle").textContent = { adjust: "Filtering", colour: "Colour", export: "Deliver" }[step] || "";
  applyFind();
  renderLayerTrack();
  updateButtons();
}

// ---------------------------------------------------------------- source
async function openPath(p) {
  if (!p) return;
  const isDisc = /\.(iso|mpls)$/i.test(p) || (await window.api.isDir(p));   // DVD (ISO, VIDEO_TS) or Blu-ray (BDMV, a playlist)
  if (isDisc) showDisc(p); else loadVideo(p);
}

async function showDisc(path) {
  status("Reading disc titles...");
  let titles;
  try { titles = await rpc("dvd_titles", { path }); } catch { return; }
  if (!titles.length) { status("No playable titles on this disc", "err"); log(`No titles found in ${path}`, "err"); return; }
  S.disc = path; S.titles = titles;
  S.titleSel = titles.reduce((b, t, k) => (t.duration > titles[b].duration ? k : b), 0);   // longest = main feature
  $("discName").textContent = basename(path);
  const tb = $("titleTable").querySelector("tbody");
  tb.innerHTML = "";
  titles.forEach((t, k) => {
    const tr = el("tr");
    const radio = el("input"); radio.type = "radio"; radio.name = "title"; radio.checked = k === S.titleSel;
    const cells = [radio, el("span", null, String(t.id)), el("span", null, hms(t.duration)), el("span", null, String(t.chapters)),
      el("span", null, t.audio.join(", ") || "-"), el("span", null, t.subs.join(", ") || "-")];
    cells.forEach((c, i) => { const td = el("td", i >= 4 ? "small" : null); td.appendChild(c); tr.appendChild(td); });
    tr.onclick = () => { S.titleSel = k; radio.checked = true; for (const r of tb.children) r.classList.remove("current"); tr.classList.add("current"); };
    tr.ondblclick = importTitle;
    if (k === S.titleSel) tr.classList.add("current");
    tb.appendChild(tr);
  });
  $("discBox").classList.remove("hidden");
  $("dropZone").classList.add("compact");
  status(`${titles.length} title${titles.length > 1 ? "s" : ""} on ${basename(path)} - choose one and import`);
}

async function importTitle() {
  const t = S.titles[S.titleSel];
  if (!t || S.busy) return;
  try {
    const out = await rpc("import", { path: S.disc, title: t.id, duration: t.duration });
    $("discBox").classList.add("hidden");
    await loadVideo(out);
  } catch { /* logged */ }
}

async function loadVideo(path) {
  const norm = (p) => String(p).replace(/\\/g, "/").toLowerCase();
  const open = SOURCES.find((x) => norm(x.path) === norm(path) && x.state);
  if (open && !(S.info && norm(S.info.path) === norm(path))) { switchSource(open.path); return; }
  sourceStash();                                         // the source open now stays in Media with its settings
  try {
    status(`Opening ${basename(path)}...`);
    const r = await rpc("probe", { path });
    S.info = r.info; S.detected = r.episodes.map((e) => e.slice()); S.episodes = r.episodes.map((e) => e.slice());
    S.checked = new Set(S.episodes.map((_, k) => k)); S.current = 0; S.queue = {};
    S.order = r.info.order; S.pal_mode = "auto";
    S.out_dir = dirname(path) + "\\restored";
    S.analyzed = false; S.auto = {}; S.frame = null; S.total = 0;
    { const rng = $("rngPos"); rng.value = Math.round(rng.max * 0.1); }   // start near the beginning, not at the
                                                                          // previous video's slider position
    pictureLoad(pictureFresh());                 // picture settings start from the defaults for every new source
    S.epSet = S.episodes.map(() => null);
    resetCleanup();
    S.layers = []; S.layerSel = -1; S.tracks = 1; S.trackOff = {}; S.src = null; S.vfrAuto = null;   // layers belong to this file
    S.gradeAll = structuredClone(GRADE0);
    S.levels = null; S.aspect = "auto";
    const restored = autosaveRestore(path, S.info.frames);
    if (restored) { sync(); log("Your auto-saved settings for this source came back (View > Auto-save settings).", "ok"); }
    if (S.script) {                              // an edited script belongs to the source it was written for
      S.script = null; refreshScript(true);
      log("Your edited script was for the previous source: the new one starts from its settings again (Script step).", "warn");
    }
    $("discBox").classList.add("hidden");
    $("dropZone").classList.add("hidden");
    $("srcCard").classList.remove("hidden"); $("epBox").classList.remove("hidden");
    $("anResults").classList.add("hidden");
    showSourceInfo(); renderEpisodes(); sync();
    log(`Opened ${path}: ${S.episodes.length} episode(s)`, "ok");
    status(S.info.image ? `Opened ${basename(path)} (${S.info.width} × ${S.info.height} image). Next: Filtering.`
                        : `Opened ${basename(path)} - ${S.episodes.length} episode${S.episodes.length > 1 ? "s" : ""}. Next: Filtering.`, "ok");
    $("placeholder").classList.add("hidden");
    S.bookmarks = []; S.snaps = [];
    sourceRegister();
    schedulePreview();
  } catch { /* logged */ }
}

function showSourceInfo() {
  const i = S.info, box = $("srcBadges");
  $("srcName").textContent = basename(i.path);
  $("srcName").title = i.path;
  box.innerHTML = "";
  const add = (text, cls) => box.appendChild(el("span", "badge " + (cls || ""), text));
  $("chapBox").classList.toggle("hidden", !!i.image || !(i.chapters || []).length);
  $("epBox").classList.toggle("hidden", !!i.image);            // a still has no episodes or chapters
  document.body.classList.toggle("image-mode", !!i.image);    // no frame stepping, timeline, play or trim for a still
  if (i.image) { S.frame = null; $("rngPos").max = 1000; $("rngPos").value = 500; }   // the middle copy: temporal stages see both sides
  if (i.image) {                                                // what matters for a picture, not video timing
    const m = i.image_info || {};
    const g = (a, b) => (b ? g(b, a % b) : a), d = g(i.width, i.height);
    const ratio = i.width / d <= 50 ? `${i.width / d}:${i.height / d}` : `${(i.width / i.height).toFixed(2)}:1`;
    add(`Image${m.format ? " · " + m.format : ""}`, "ntsc");
    add(`${i.width} × ${i.height} px`);
    add(`${ratio} ${i.width > i.height ? "landscape" : i.width < i.height ? "portrait" : "square"}`);
    add(`${(i.width * i.height / 1e6).toFixed(1)} MP`);
    if (m.colour) add(`${m.colour}${m.bits ? ", " + m.bits + "-bit" : ""}${m.alpha ? " + transparency" : ""}`);
    if (m.bytes) add(m.bytes >= 1048576 ? `${(m.bytes / 1048576).toFixed(1)} MB` : `${Math.round(m.bytes / 1024)} KB`);
    return;
  }
  const rate = i.fps_num / i.fps_den, r = Math.abs(rate - Math.round(rate)) < 0.01 ? String(Math.round(rate)) : rate.toFixed(3).replace(/0+$/, "");
  if (i.progressive) add(`Progressive ${i.vfr_hint ? "~" + r + " fps (variable)" : r + " fps"}`, "ntsc");
  else add(i.pal ? "PAL 25 fps" : "NTSC 29.97 fps", i.pal ? "pal" : "ntsc");
  add(`${i.width}x${i.height}`); add(i.dar > 1.5 ? "16:9" : "4:3"); add(tc(i.frames));
  if (!i.progressive) add(i.order === "bff" ? "bottom field first" : "top field first");
  add(`${i.chapters.length} chapters`);
  if (i.audio.length) add("audio " + i.audio.map((a) => a.lang).join(", "));
  if (i.subs.length) add(`subs ${i.subs.map((s) => s.lang).join(", ")}`);
}

function resetCleanup() {
  Object.assign(S, { chroma_unmix: "off", crawl: "off", rainbow: "off", composite: "off", ghost: "off", ghost_taps: [], halo_rx: 0,
                     ghost_strength: 1, sx: 0, sy: 0, deblend: "off", rainbow_fill: false, deblock: "off", crop: { l: 0, r: 0, t: 0, b: 0 } });
}

// ---------------------------------------------------------------- episodes
function renderEpisodes() {
  const tb = $("epTable").querySelector("tbody");
  tb.innerHTML = "";
  S.episodes.forEach(([a, b], k) => {
    const tr = el("tr", k === S.current ? "current" : null);
    const cb = el("input"); cb.type = "checkbox"; cb.checked = S.checked.has(k);
    cb.onclick = (e) => { e.stopPropagation(); cb.checked ? S.checked.add(k) : S.checked.delete(k); epCount(); sync(); };
    const mk = (val, idx) => {
      const inp = el("input"); inp.type = "text"; inp.value = tc(val);
      inp.onclick = (e) => e.stopPropagation();
      inp.onchange = () => {
        const v = parseTc(inp.value);
        if (!isNaN(v)) { S.episodes[k][idx] = Math.max(0, Math.min(v, S.info.frames)); renderEpisodes(); schedulePreview(); }
      };
      return inp;
    };
    // chapter boxes: start = chapter starting at the start time, end = chapter ending at the end time
    const cf = chapFrames(), nch = cf.length;
    const chapEnd = (i) => (i + 1 < nch ? cf[i + 1] : S.info.frames);
    const near = (x, y) => Math.abs(x - y) <= 2;
    const mkc = (idx) => {
      const inp = el("input"); inp.type = "text"; inp.className = "chap mono";
      const cur = idx === 0 ? cf.findIndex((f) => near(f, a)) : cf.findIndex((_, i) => near(chapEnd(i), b));
      inp.value = cur >= 0 ? String(cur + 1) : "";
      inp.placeholder = "–";
      inp.disabled = nch < 2;
      inp.title = nch < 2 ? "No chapters on this source" : (idx === 0 ? "Start at the beginning of chapter …" : "End after chapter …") + ` (1-${nch})`;
      const apply = (n) => {
        if (isNaN(n) || n < 1 || n > nch) { renderEpisodes(); return; }
        const f = idx === 0 ? cf[n - 1] : chapEnd(n - 1);
        const ep = S.episodes[k];
        if (idx === 0 && f >= ep[1] - 2) { status(`Chapter ${n} starts after this episode's end.`, "err"); renderEpisodes(); return; }
        if (idx === 1 && f <= ep[0] + 2) { status(`Chapter ${n} ends before this episode's start.`, "err"); renderEpisodes(); return; }
        ep[idx] = f;
        if (k === S.current) { S.frame = idx === 0 ? 0 : null; S.total = 0; }
        renderEpisodes(); schedulePreview(); scheduleScript();
      };
      inp.onclick = (e) => e.stopPropagation();
      inp.onchange = () => apply(parseInt(inp.value, 10));
      inp.onkeydown = (e) => {
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          const base = parseInt(inp.value, 10) || (idx === 0 ? 1 : nch);
          apply(Math.max(1, Math.min(nch, base + (e.key === "ArrowUp" ? 1 : -1))));
          setTimeout(() => { const again = $("epTable").querySelectorAll("tbody tr")[k]?.querySelectorAll("input.chap")[idx]; again && again.focus(); }, 0);
        }
      };
      return inp;
    };
    const cells = [cb, el("span", null, String(k + 1)), mk(a, 0), mkc(0), mk(b, 1), mkc(1), el("span", null, hmsms(span(a, b)).slice(2, -4))];
    cells.forEach((c) => { const td = el("td"); td.appendChild(c); tr.appendChild(td); });
    tr.onclick = () => selectEpisode(k);
    tb.appendChild(tr);
  });
  epCount(); sync(); trimInfo(); renderChapters();
}

// ---------------------------------------------------------------- chapters
const chapFrames = () => (S.info && S.info.chapters ? S.info.chapters.map((t) => timeFrame(t)) : []);
const span = (a, b) => frameTime(b) - frameTime(a);              // seconds between two frames
function chapRange(a, b) {                       // "3-7" = chapters (1-based) that start inside [a, b)
  const cf = chapFrames();
  const ins = cf.map((f, i) => [f, i]).filter(([f]) => f >= a - 2 && f < b - 2).map(([, i]) => i + 1);
  if (!ins.length) return "-";
  return ins.length === 1 ? String(ins[0]) : `${ins[0]}–${ins[ins.length - 1]}`;
}
function renderChapters() {
  const box = $("chapBox"), cf = chapFrames();
  box.classList.toggle("hidden", cf.length < 2);
  if (cf.length < 2) return;
  const tb = $("chapTable").querySelector("tbody"); tb.innerHTML = "";
  const ep = S.episodes[S.current] || [0, 0], end = S.info.frames;
  $("chapHint").textContent = `${cf.length} · episode ${S.current + 1} highlighted`;
  cf.forEach((f, i) => {
    const nx = i + 1 < cf.length ? cf[i + 1] : end;
    const tr = el("tr", (f >= ep[0] - 2 && f < ep[1] - 2 ? "in" : "") + (Math.abs(f - ep[0]) <= 2 ? " edge" : ""));
    const t = el("span", "tlink mono", tc(f).slice(0, -4)); t.title = "Preview this chapter";
    t.onclick = () => previewChapter(f);
    const btns = el("td", "btns");
    const b = (txt, title, fn) => { const x = el("button", null, txt); x.title = title; x.onclick = (e) => { e.stopPropagation(); fn(); }; btns.appendChild(x); };
    b("Start", "The selected episode starts at this chapter", () => setEpisodeEdge(0, f));
    b("End", "The selected episode ends after this chapter", () => setEpisodeEdge(1, nx));
    b("Split", "A new episode starts at this chapter", () => splitAt(f));
    for (const c of [el("span", "mono", String(i + 1)), t, el("span", "mono", hms(span(f, nx)).replace(/^0:0?/, ""))]) {
      const td = el("td"); td.appendChild(c); tr.appendChild(td);
    }
    tr.appendChild(btns);
    tb.appendChild(tr);
  });
}
function setEpisodeEdge(which, f) {
  if (!S.episodes.length) S.episodes = [[0, S.info.frames]];
  const ep = S.episodes[S.current];
  if (which === 0 && f >= ep[1] - 2) { status("Start must be before the episode's end.", "err"); return; }
  if (which === 1 && f <= ep[0] + 2) { status("End must be after the episode's start.", "err"); return; }
  ep[which] = f; S.frame = which === 0 ? 0 : null; S.total = 0;
  renderEpisodes(); schedulePreview(); scheduleScript();
}
function splitAt(f) {
  const k = S.episodes.findIndex(([a, b]) => f > a + 2 && f < b - 2);
  if (k < 0) { status("This chapter is already where an episode starts or outside every episode.", "err"); return; }
  const [a, b] = S.episodes[k];
  epSetAlign();
  S.epSet[S.current] = pictureSnap();                            // the episode shown keeps what is set now
  const own = S.epSet[k];
  S.episodes.splice(k, 1, [a, f], [f, b]);
  S.epSet.splice(k, 1, own ? structuredClone(own) : null, own ? structuredClone(own) : null);   // both halves keep the settings
  S.checked = new Set([...S.checked].flatMap((i) => (i < k ? [i] : i === k ? [k, k + 1] : [i + 1])));
  S.current = k + 1; S.frame = 0; S.total = 0;
  pictureLoad(S.epSet[S.current] || pictureFresh());
  sync();
  renderEpisodes(); schedulePreview(); scheduleScript();
  log(`Episode ${k + 1} split at ${tc(f)}: now episodes ${k + 1} and ${k + 2}`, "ok");
}
function previewChapter(f) {
  const k = S.episodes.findIndex(([a, b]) => f >= a && f < b);
  if (k < 0) return;
  togglePlay(false);
  if (k !== S.current) { S.current = k; renderEpisodes(); }
  const [a, b] = S.episodes[k];
  S.frame = null; S.total = 0;
  $("rngPos").max = 1000; $("rngPos").value = Math.round(1000 * (f - a) / Math.max(1, b - a));
  schedulePreview();
}
function epCount() {
  $("epCount").textContent = S.episodes.length ? `${S.checked.size} of ${S.episodes.length} to export` : "";
  $("chkAllEp").checked = S.episodes.length > 0 && S.checked.size === S.episodes.length;
}
function selectEpisode(k) {
  if (k === S.current || k < 0 || k >= S.episodes.length) return;
  togglePlay(false);
  epSetAlign();
  S.epSet[S.current] = pictureSnap();
  S.current = k; S.frame = null; S.total = 0; S.src = null; S.vfrAuto = null; S.rangeSel = null;
  pictureLoad(S.epSet[k] || pictureFresh());
  if (!S.epSet[k]) log(`Episode ${k + 1}: its own settings start from the defaults (Copy to all episodes copies the current ones).`);
  renderEpisodes(); sync(); schedulePreview(); scheduleScript();
}
function copySettingsToAll() {
  epSetAlign();
  const snap = pictureSnap();
  S.epSet = S.episodes.map(() => structuredClone(snap));
  log(`The settings of episode ${S.current + 1} now apply to all ${S.episodes.length} episodes (each can still be changed on its own).`, "ok");
  status(`Settings copied to all ${S.episodes.length} episodes.`, "ok");
}
// the picture settings of episode k in the backend's form (export)
function pictureState(k) {
  const live = pictureSnap();
  if (k !== S.current) pictureLoad(S.epSet[k] || pictureFresh());
  const st = backendState([k]), out = {};
  for (const key of ["mode", "order", "pal_mode", "comb", "ivtc_repair", "chroma_unmix", "aa", "deblend", "crawl", "rainbow", "composite", "rainbow_fill",
                     "deblock", "ghost", "ghost_strength", "ghost_taps", "halo_rx", "halo_method", "sx", "sy", "bleed", "warp", "neutral",
                     "stab", "stab_level", "stab_pos", "repair_order", "removed_steps", "tfix_strength", "crop", "aspect", "aspect_custom", "size", "custom_w", "custom_h", "custom_lock",
                     "kernel", "kernel_b", "kernel_c", "sharp", "upscale", "grade", "layers", "vs_filters", "vs_filters_pos", "vs_groups"]) out[key] = st[key];
  if (k !== S.current) pictureLoad(live);
  return out;
}

// ---------------------------------------------------------------- analyze
async function analyze() {
  try {
    status("Analyzing...");
    const r = await rpc("analyze", { state: backendState([S.current]) });
    const auto = {};
    if ("crawl" in r) Object.assign(auto, { crawl: r.crawl, rainbow: r.rainbow, composite: r.composite || "off" });
    if (r.pal) Object.assign(auto, { pal_mode: r.pal.mode, order: r.pal.order });
    Object.assign(auto, { ghost: r.ghost === "auto" ? "auto" : "off", sx: r.shift_x, sy: r.shift_y,
                          bleed: `${r.method || "off"}|${r.strength || "normal"}` });
    if ("deblock" in r) auto.deblock = r.deblock;
    if (S.ai_model === "field" || S.ai_model === "uni") {   // FieldNet / UniNet: their MPEG / Halos dials follow the disc
      if ("deblock" in r) auto.uni_mpeg = MPEG_DIAL[r.deblock] ?? 1;
      auto.uni_halo = haloDial(r);
    }
    if ("chroma_unmix" in r) auto.chroma_unmix = r.chroma_unmix;
    if (r.warp) auto.warp = r.warp;
    if (r.stab) Object.assign(auto, { stab: r.stab, stab_level: r.stab_level || "normal" });
    // crop and colour have their own Auto buttons: keep what those found
    for (const [k, v] of Object.entries(S.auto)) if (k.startsWith("crop.") || k.startsWith("grade.")) auto[k] = v;
    S.auto = auto;
    for (const [k, v] of Object.entries(auto)) setv(k, v);
    S.ghost_taps = r.ghost_taps || []; S.halo_rx = r.halo_rx || 0;
    S.analyzed = true;
    showResults(r);
    sync(); schedulePreview();
    log("Analysis applied: " + Object.entries(auto).map(([k, v]) => `${k} ${v}`).join(", "), "ok");
    status("Analysis done - check the preview, then Adjust. Colour and Crop have their own Auto buttons.", "ok");
  } catch { /* logged */ }
}

// Adjust: one Auto per group (Analyze step removed). Values are suggestions - the notice says so.
// size of an artifact measured in 8-bit levels (RMS of the pattern itself, noise and grain excluded)
const lv = (v) => (v == null ? "not measurable" : `${(+v).toFixed(2)} levels`);
// FieldNet dials from the measurements: MPEG removal only as far as the disc has blocking, halo removal in proportion
// to the measured halo (its removal also takes thin drawn strokes next to outlines: Courage frame 1575)
const MPEG_DIAL = { off: 0, light: 0.6, normal: 1 };
const haloDial = (r) => (r.halo_rx ? Math.min(1, Math.round((r.halo || 0) / 0.5 * 20) / 20) : 0);
const AUTO_KEYS = {
  frames: (r) => (r.pal_mode ? { pal_mode: r.pal_mode, order: r.order } : Object.assign({ chroma_unmix: r.chroma_unmix, mode: r.mode, comb: r.comb, ivtc_repair: r.ivtc_repair, aa: r.aa }, r.order_measured ? { order: r.order_measured } : {})),
  composite: (r) => ({ crawl: r.crawl, rainbow: r.rainbow, composite: r.composite || "off" }),
  block: (r) => (S.ai_model === "field" || S.ai_model === "uni" ? { deblock: r.deblock, uni_mpeg: MPEG_DIAL[r.deblock] ?? 1 } : { deblock: r.deblock }),
  ghost: (r) => Object.assign({ ghost: r.ghost_on || (r.halo_rx ? "auto" : "off") }, (S.ai_model === "field" || S.ai_model === "uni") ? { uni_halo: haloDial(r) } : {}),
  chroma: (r) => Object.assign({ sx: r.shift_x, sy: r.shift_y, bleed: `${r.method || "off"}|${r.strength || "normal"}` }, r.warp ? { warp: r.warp } : {}),
  stab: (r) => ({ stab: r.stab, stab_level: r.stab_level || "normal" }),
};
function autoText(kind, r) {
  const pc = (v) => (v == null ? "?" : `${v}`);
  const T = {
    frames: () => r.pal_mode ? `Telecine & interlacing: PAL ${r.pal_mode}, ${r.order === "bff" ? "bottom" : "top"} field first (combed ${(100 * r.raw_combed).toFixed(1)} % -> ${(100 * r.matched_combed).toFixed(1)} % after field matching).`
      : `Telecine & interlacing: ${r.mode === "bob" ? "60p deinterlace - interlaced video (every field is its own moment; inverse telecine would drop half the motion)"
          : r.mode === "vfr" ? `VFR - ${r.video_sections} video-rate section(s) found (captions / titles at 30p)` : "CFR 23.976 - film only, no video-rate sections"} `
        + `(${pc(r.video_share)} % of moving cycles). Unmatched frames ${pc(r.unmatched_share)} % (combing ${pc(r.comb_strength)} % of the picture) -> combing repair ${r.comb}, anti-aliasing ${r.aa}`
        + `${r.line_doubled > 40 ? ` (source looks line-doubled: ${r.line_doubled} %)` : ""}. ICP ${r.chroma_unmix === "on" ? "found" : "not found"}.`
        + (r.order_measured ? ` Field order from the picture: ${r.order_measured === "bff" ? "bottom" : "top"} first${S.info && S.info.order !== r.order_measured ? " (the file says otherwise - corrected)" : ""}.` : ""),
    composite: () => (r.note ? r.note + " " : "") + `Composite: dot crawl ${r.crawl} (${lv(r.crawl_level)}${r.carrier ? `, carrier ${r.carrier} cycles/px` : ""}), rainbows ${r.rainbow} (${lv(r.rainbow_level)}).`,
    block: () => `Compression: the 8-px block grid adds a step of ${lv(r.block_step)} (worst tenth of frames; plainly visible from 0.9) - MPEG repair ${r.deblock}.`,
    ghost: () => r.halo_rx ? `Halos: ring ${r.halo_pos} px from edges, strength ${(r.halo || 0).toFixed(2)} - removal on.` : `Halos: none found (${(r.halo || 0).toFixed(2)}).`,
    chroma: () => r.measured ? `Colour: sits x ${sign(r.measured.x)} / y ${sign(r.measured.y)} px off the outlines, edges ${r.measured.bleed.toFixed(2)} px wider - shift ${sign(r.shift_x)} / ${sign(r.shift_y)}, bleed fix ${r.method || "off"}.` : "Colour: no usable edges found.",
    stab: () => r.stab === "ustab" ? `Stabilizer: Universal ${r.stab_level} - steadies flicker, boil and wobble${r.ai_cleaner ? " the frame-by-frame cleanup leaves" : ` (dot crawl ${lv(r.crawl_level)})`}; fades, cuts and flashes are never dragged.`
      : r.stab === "chain" ? `Stabilizer: Steady chain ${r.stab_level} - the recurrent cleanup still leaves flicker on fine still detail (Courage: door 1.5 -> 0.16, lettering shimmer 0.8 -> 0.04).`
      : r.stab !== "off" ? `Stabilizer: dot crawl ${lv(r.crawl_level)} - ${{ustab: "Universal", gtfix: "Temporal fix (guided)", ttemp: "TTempSmooth", chain: "Steady chain"}[r.stab] || r.stab} ${r.stab_level}.`
      : r.stab === "chain" ? `Stabilizer: Steady chain ${r.stab_level} - the recurrent cleanup still leaves flicker on fine still detail.`
      : r.stable_net ? "Stabilizer: off - StableNet is steady by itself; a stabilizer on top only softens."
      : r.ai_cleaner ? `Stabilizer: off - the AI cleanup handles the dot crawl (${lv(r.crawl_level)}); too little left over for a stabilizer.`
      : "Stabilizer: no dot crawl to steady - off.",
  };
  try { return T[kind](); } catch (e) { return `${kind}: done`; }
}
async function autoGroup(kind, btn) {
  if (S.busy || !S.info) return;
  btn.classList.add("running"); const old = btn.textContent; btn.textContent = "…";
  try {
    const r = await rpc("detect", { kind: kind === "ghost" ? "ghost" : `auto_${kind}`, state: backendState([S.current]) });
    const vals = AUTO_KEYS[kind](r);
    for (const [k, v] of Object.entries(vals)) { if (v === undefined) continue; S.auto[k] = v; setv(k, v); }
    if (kind === "ghost") { S.ghost_taps = r.taps || []; S.halo_rx = r.halo_rx || 0; }
    S.analyzed = true;
    const msg = autoText(kind, r) + " (suggestion - check the preview)";
    repairFound(kind, autoText(kind, r));
    $("autoResult").textContent = msg; log(msg, "ok"); status(msg, "ok");
    sync(); schedulePreview(); scheduleScript();
  } catch { /* logged */ }
  btn.classList.remove("running"); btn.textContent = old;
}

const WHEEL0 = [0, 0, 0];
const wheelSet = (w) => Array.isArray(w) && w.some((x) => Math.abs(x) > 1e-4);

// Auto colour's findings, one line each: [id, title, what was found, what it sets]; COLOUR_AUTO_KEYS = the grade keys each owns
const COLOUR_AUTO_KEYS = { levels: ["black", "white"], tone: ["contrast", "gamma", "shadows"], white: ["gain_w", "temp", "tint"],
                           black: ["lift"], fade: ["sat"], shadow: ["dark_sat"], hue: ["hue"], gamut: ["gamut"] };
function autoItems(r) {
  const g = r.grade || {}, items = [];
  const lv = g.black !== 16 || g.white !== 235;
  items.push(["levels", "Levels", `picture spans ${r.level_black} - ${r.level_white} (DVD: 16 - 235)`,
    lv ? `black ${g.black}, white ${g.white}` : null]);
  const ref = r.ref_frames != null, same = !!r.ref_same;
  const other = "the reference shows other scenes - for this, use the same episode from the better release (or screenshots of the same moments)";
  if (ref) items.push(["refinfo", "Reference", same ? `same scenes: ${r.ref_pairs || r.ref_frames} frames matched picture for picture`
    : `other scenes (${r.ref_match || 0} % of the frames found): only a clear fade (20 %+) is matched`, null]);
  const tn = (g.gamma ?? 1) !== 1 || (g.shadows ?? 0) !== 0 || (g.contrast ?? 1) !== 1;
  const tset = [(g.contrast ?? 1) !== 1 ? `contrast ${g.contrast}` : "", (g.gamma ?? 1) !== 1 ? `midtones ${g.gamma}` : "",
                (g.shadows ?? 0) !== 0 ? `shadow lift ${g.shadows}` : ""].filter(Boolean).join(", ");
  const est = r.prior_gamma != null, guess = " (learned estimate - a reference measures it)";
  items.push(["tone", "Tone", !ref ? (!est ? "needs a reference to judge (contrast, midtones, shadow lift)"
      : tn ? `midtones look ${(g.gamma ?? 1) > 1 ? "too dark" : "too bright"} for a good transfer` + guess : "midtones look right" + guess) : !same ? other
    : tn ? "brightness curve differs from the reference" : "matches the reference", tn ? tset : null]);
  let wv;
  if (r.wb_u == null) wv = "too few white areas to judge";
  else if (r.monochrome) wv = "one tinted look - the art, left alone";
  else if (wheelSet(g.gain_w)) wv = `tinted (Cb ${sign(r.wb_u)} / Cr ${sign(r.wb_v)}) in ${r.wb_consistent} % of the frames - a transfer cast`;
  else if (Math.hypot(r.wb_u, r.wb_v) < 1.5) wv = "neutral";
  else wv = "change from shot to shot - the art, left alone";
  // few different shots (a short clip, a still): a cast there may be that scene's own look - say so
  const few = r.shots != null && r.shots < 8 ? ` · only ${r.shots} shot${r.shots === 1 ? "" : "s"} measured: if this scene is meant to look tinted, untick` : "";
  items.push(["white", "Whites", wv + (wheelSet(g.gain_w) ? few : ""), wheelSet(g.gain_w) ? "Highlights wheel" : null]);
  items.push(["black", "Blacks", r.bk_u == null ? "too few dark areas to judge" : wheelSet(g.lift) ? `line art tinted (Cb ${sign(r.bk_u)} / Cr ${sign(r.bk_v)})` + few : "neutral",
    wheelSet(g.lift) ? "Shadows wheel" : null]);
  const sf = (g.sat ?? 1) !== 1;
  items.push(["fade", "Saturation", ref ? (sf ? `colour ${Math.round(100 / (g.sat || 1))} % as strong as the reference's` : "matches the reference")
    : sf ? (r.prior_used && est ? `colour ${(g.sat ?? 1) > 1 ? "weaker" : "stronger"} than good transfers look` + guess : `faded: strongest colours ${r.sat_peak} (good transfers 0.17+)`)
    : est ? "colour strength looks right" + guess : "not clearly faded (a reference judges a moderate fade)", sf ? `saturation ${g.sat}` : null]);
  items.push(["shadow", "Shadow colour", !ref ? "needs a reference to judge" : !same ? other : (g.dark_sat || 0) > 0 ? "dark tones weaker in colour than the reference's"
    : "matches the reference", (g.dark_sat || 0) > 0 ? `shadow colour ${g.dark_sat}` : null]);
  items.push(["hue", "Hue", !ref ? (!est ? "needs a reference to judge" : (g.hue || 0) !== 0 ? `colours look turned by ${-g.hue}°` + guess : "colours look right" + guess) : !same ? other : (g.hue || 0) !== 0 ? `every colour turned ${-g.hue}° (NTSC phase)` : "matches the reference",
    (g.hue || 0) !== 0 ? `hue ${g.hue > 0 ? "+" : ""}${g.hue}°` : null]);
  items.push(["gamut", "Clipped colour", r.gamut_out >= 1 ? `${r.gamut_out} % beyond the screen range` : "within the screen range", g.gamut && g.gamut !== "off" ? g.gamut : null]);
  return items;
}

// The measured grade, as ticked and at the chosen strength, into the base grade (only the keys of `which`)
function applyAuto(which = Object.keys(COLOUR_AUTO_KEYS)) {
  const r = S.levels; if (!r || !r.grade) return;
  const g = r.grade, s = +S.autoStrength;
  const mix = (v, d) => +(d + s * (v - d)).toFixed(3);
  const val = {
    black: Math.round(mix(g.black ?? 16, 16)), white: Math.round(Math.min(255, mix(g.white ?? 235, 235))),
    gain_w: (g.gain_w || WHEEL0).map((x) => mix(x, 0)), temp: 0, tint: 0, lift: (g.lift || WHEEL0).map((x) => mix(x, 0)),
    sat: +mix(g.sat ?? 1, 1).toFixed(2), gamut: g.gamut || "off",
    contrast: +mix(g.contrast ?? 1, 1).toFixed(3), gamma: +mix(g.gamma ?? 1, 1).toFixed(2), shadows: +mix(g.shadows ?? 0, 0).toFixed(2),
    dark_sat: +mix(g.dark_sat ?? 0, 0).toFixed(2), hue: Math.round(mix(g.hue ?? 0, 0)),
  };
  for (const id of which) for (const k of COLOUR_AUTO_KEYS[id]) {
    const v = S.autoUse[id] ? val[k] : structuredClone(GRADE0[k]);
    S.gradeAll[k] = v; S.auto[`grade.${k}`] = v;
  }
  sync(); syncColourUi(); schedulePreview(); scheduleScript();
}

function findingsUi() {
  const box = $("autoFindings"), r = S.levels;
  box.innerHTML = "";
  $("autoStrengthRow").classList.toggle("hidden", !(r && r.grade));
  if (!r || !r.grade) return;
  for (const [id, title, found, sets] of autoItems(r)) {
    const row = el("label", "finding" + (sets ? "" : " fine") + (id === "refinfo" ? " info" : "") + (found.startsWith("needs a reference") ? " na" : "") + (sets && !S.autoUse[id] ? " off" : ""));
    const cb = el("input"); cb.type = "checkbox"; cb.checked = !!sets && !!S.autoUse[id]; cb.disabled = !sets;
    cb.onchange = () => { S.autoUse[id] = cb.checked; applyAuto([id]); };
    row.append(cb, el("b", null, title), el("span", "found", found), el("span", "sets", sets ? `→ ${sets}` : "✓ fine"));
    row.title = sets ? "Untick to leave this part out" : "Nothing to correct";
    box.appendChild(row);
  }
}

// Colour step: measure the source and set the grade auto colour solved (own button, not part of Analyze)
async function autoColour(wbOnly = false) {
  try {
    status(wbOnly ? "Measuring the colour of the whites and blacks..." : "Measuring levels, white and black colour, fade and clipped colour...");
    const r = await rpc("detect", { kind: "levels", state: backendState([S.current]) });
    if (!r.grade) { status("Nothing to measure (no picture found).", "err"); return; }
    S.levels = r;
    for (const id of Object.keys(COLOUR_AUTO_KEYS)) S.autoUse[id] = !wbOnly || id === "white" || id === "black";
    refUi();
    applyAuto(wbOnly ? ["white", "black"] : undefined);
    const items = autoItems(r).filter(([id, , , sets]) => sets && S.autoUse[id]);
    const msg = items.length ? "Auto colour: " + items.map(([, t, f, sets]) => `${t.toLowerCase()} ${f} → ${sets}`).join("; ") + "."
      : "Auto colour: this source needs no correction (levels, whites, blacks and colour are right).";
    log(msg, "ok"); status(msg, "ok");
  } catch { /* logged */ }
}

// Colour step: the reference (images / video of the same show that look right)
function refUi() {
  const L = S.colourRef || [];
  $("refName").textContent = L.length ? "Reference: " + L.map(basename).join(", ")
    : "no reference - saturation, tone and hue are only corrected when clearly wrong";
  $("refName").title = L.join("\n");
  $("btnRefClear").classList.toggle("hidden", !L.length);
}
function initRef() {
  $("btnRefAdd").onclick = async () => {
    const p = await window.api.openRefs();
    if (p && p.length) { S.colourRef = p; refUi(); status("Reference set - press Auto colour to match it.", "ok"); }
  };
  $("btnRefClear").onclick = () => { S.colourRef = []; refUi(); status("Reference removed - press Auto colour again to measure without it."); };
  refUi();
}

// Colour step: fold / unfold a section by clicking its title (remembered between sessions)
function buildFolds() {
  let st = {};
  try { st = JSON.parse(localStorage.getItem("colourFolds") || "{}"); } catch { /* fresh */ }
  for (const g of $$(".group.fold")) {
    if (g.closest("#gradeDock")) { g.classList.remove("closed"); continue; }   // the panel under the preview: always open
    const id = g.dataset.fold, h = g.querySelector(".group-head h3");
    g.classList.toggle("closed", id in st ? !!st[id] : g.dataset.closed === "1");
    h.title = "Fold / unfold";
    h.onclick = () => {
      g.classList.toggle("closed");
      st[id] = g.classList.contains("closed");
      localStorage.setItem("colourFolds", JSON.stringify(st));
      if (!st[id]) { syncColourUi(); if (id === "scopes") requestScopes(); }
    };
  }
}

// Crop & size step: find the black borders (own button, not part of Analyze)
async function autoCrop() {
  try {
    status("Looking for black borders...");
    const c = await rpc("detect", { kind: "crop", state: backendState([S.current]) });
    for (const k of "lrtb") { S.auto[`crop.${k}`] = c[k]; S.crop[k] = c[k]; }
    sync(); schedulePreview();
    const msg = c.l || c.r || c.t || c.b ? `Borders: left ${c.l}, right ${c.r}, top ${c.t}, bottom ${c.b} px - crop set to ${S.info.width - c.l - c.r}x${S.info.height - c.t - c.b}.`
                                         : "No black borders found.";
    repairFound("crop", msg);
    log(msg, "ok"); status(msg, "ok");
  } catch { /* logged */ }
}

function showResults(r) {
  const list = $("anList");
  list.innerHTML = "";
  const label = (key, v) => { const o = S.opts[key].find((x) => (Array.isArray(x) ? x[0] : x.key) === v); return o ? (Array.isArray(o) ? o[1] : o.label) : v; };
  const row = (name, meas, set, off) => {
    const d = el("div", "res");
    d.append(el("span", "name", name), el("span", "set" + (off ? " off" : ""), set), el("span", "meas", meas));
    list.appendChild(d);
  };
  if (r.pal) row("PAL type", `combed frames ${(100 * r.pal.raw_combed).toFixed(1)}% → ${(100 * r.pal.matched_combed).toFixed(1)}% after field matching`,
                 label("pal_modes", r.pal.mode));
  if ("crawl_level" in r) {
    row("Dot crawl", `${lv(r.crawl_level)} (under 0.4 = clean)` + (r.carrier ? ` · carrier ${r.carrier} cycles/px` : ""), label("levels", r.crawl), r.crawl === "off");
    row("Rainbows", `${lv(r.rainbow_level)} · composite decode ${label("composite", r.composite || "off")}`, label("levels", r.rainbow), r.rainbow === "off");
  }
  if ("chroma_unmix" in r) row("ICP fix", `colour from a neighbouring frame in ${r.unmix_assembled} % of the frames IVTC assembles, ${r.unmix_whole} % of the others (${r.unmix_frames} assembled frames checked)`,
                                r.chroma_unmix === "on" ? "Repair" : "Off", r.chroma_unmix !== "on");
  if ("block_step" in r) row("MPEG blocking", `extra step at the block grid ${lv(r.block_step)} in the worst tenth of frames (every DVD has ~0.5; plainly visible from 0.9)`, r.deblock === "off" ? "Off" : (r.deblock === "light" ? "Light" : "Normal"), r.deblock === "off");
  if ("stab" in r) row("Temporal flicker", r.stab !== "off" ? "leftover crawl, flicker, boil and wobble are steadied over time" : "nothing to steady",
                      r.stab !== "off" ? `${{ustab: "Universal", gtfix: "Temporal fix (guided)", ttemp: "TTempSmooth", chain: "Steady chain"}[r.stab] || r.stab} ${r.stab_level}` : "Off", r.stab === "off");
  row("Halos", r.halo_rx ? `ring ${r.halo_pos} px from edges, strength ${(r.halo || 0).toFixed(2)}` : `none found (${(r.halo || 0).toFixed(2)})`,
      r.halo_rx ? "Removal on" : "Off", !r.halo_rx);
  const m = r.measured;
  row("Colour shift", m ? `colour sits x ${sign(m.x)} px, y ${sign(m.y)} px from the outlines` : "no edges found",
      `${sign(r.shift_x)} / ${sign(r.shift_y)} px`, !r.shift_x && !r.shift_y);
  row("Colour bleed", m ? `colour edges ${m.bleed.toFixed(2)} px wider than outlines` : "-", label("bleed", `${r.method || "off"}|${r.strength || "normal"}`), (r.method || "off") === "off");
  if (r.crop) {
    const c = r.crop;
    row("Borders", `left ${c.l}, right ${c.r}, top ${c.t}, bottom ${c.b} px`, `crop to ${S.info.width - c.l - c.r}x${S.info.height - c.t - c.b}`, !(c.l || c.r || c.t || c.b));
  }
  $("anResults").classList.remove("hidden");
}

async function collectReport() {
  try {
    status("Collecting report (about a minute)...");
    const r = await rpc("report", { state: backendState([S.current]), log: $("log").innerText.slice(-20000) });
    log(`Report ready: ${r.path} (${r.mb} MB). Send this zip file.`, "ok");
    status(`Report saved (${r.mb} MB) - send the zip that is now selected in Explorer.`, "ok");
    window.api.showItem(r.path);
  } catch { /* logged */ }
}

// ---------------------------------------------------------------- preview
let previewTimer = null, previewSeq = 0;
function schedulePreview(delay = 120) {          // 120 ms: slider drags (a render takes ~90 ms); one-off clicks pass less
  histArm();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(renderPreview, delay);
  vsviewFollow();
}
// the open vsview follows every change: its job file is rewritten and it reloads itself (same tab and frame)
let vsviewTimer = null;
function vsviewFollow() {
  if (!S.vsviewLive || S.current == null) return;
  clearTimeout(vsviewTimer);
  vsviewTimer = setTimeout(() => rpc("vsview_update", { state: vsviewState() }).catch(() => {}), 700);
}
function previewPos() {
  if (S.total > 1 && S.frame != null) return S.frame / (S.total - 1);
  const r = $("rngPos");
  return r.value / r.max;
}
// ---------------------------------------------------------------- play + trim
const PLAY = { on: false, t: 0, n: 0, last: 0 };
function vfrButton() {
  const b = $("btnVfrPrev"), vfr = S.mode === "vfr" && S.info && !S.info.pal && !isProgressive();
  b.classList.toggle("hidden", !S.info);
  const near = (r) => Math.abs(S.rate - r) < r * 0.01;           // ms-rounded timestamps: 23.976 can read 23.81
  const rate = S.rate ? " · " + (near(23.976) ? "24p" : near(29.97) ? "30p" : near(25) ? "25p" : near(59.94) ? "60p" : near(50) ? "50p" : S.rate.toFixed(2)) : "";
  const a = S.vfrAuto || {};
  let t, failed = false;
  if (!S.info) t = "";
  else if (!vfr) t = (S.info.pal ? "PAL preview" : isProgressive() ? "Progressive" : S.mode === "bob" ? "60p deinterlaced" : "CFR preview") + rate;
  else if (S.vfrPrev) t = "VFR preview ✓" + rate;
  else if (a.state === "failed") { t = "VFR analysis failed · retry"; failed = true; }
  else if (a.state === "running") t = `VFR analysis ${a.progress ? Math.round(100 * a.progress) + " %" : "…"} · 24p meanwhile`;
  else { t = "VFR not analyzed · 24p preview"; failed = true; }
  b.textContent = t;
  b.title = !vfr ? "The preview shows the export's frames (constant frame rate; no VFR analysis runs)."
    : "VFR: until this episode is analyzed (Telecine & interlacing › Analyze, or click here) the preview shows 24p; then the export's own frames and frame rates (30 fps video parts such as scrolling captions play smoothly). The export reuses the analysis, or runs it itself.";
  b.classList.toggle("primary", false);
  b.disabled = !failed || S.busy;
  telButton();
}
// Telecine & interlacing › Analyze: 1. progressive or interlaced / telecined (the picture), 2. Auto (VFR / CFR / 60p, field
// order, ICP fix, combing / unmatched-frame repair; PAL: its type), 3. the VFR analysis when it is VFR
const TEL = { step: null, done: {} };
function telKey() { return `${S.info ? S.info.path : ""}#${S.current}`; }
function telButton() {
  const an = $("btnVfrAnalyze");
  if (!an) return;
  an.classList.toggle("hidden", !S.info || !!(S.info && S.info.image));
  const vfr = S.mode === "vfr" && S.info && !S.info.pal && !isProgressive();
  const a = S.vfrAuto || {}, vfrRun = vfr && a.state === "running";
  const done = TEL.done[telKey()] && (!vfr || S.vfrPrev || a.state === "ready");
  an.textContent = TEL.step === "scan" ? "Checking the picture…" : TEL.step === "auto" ? "Detecting…"
    : vfrRun ? `Step 3 · VFR pass ${a.progress ? Math.round(100 * a.progress) + " %" : "…"}` : done ? "Analyzed ✓" : "Analyze";
  an.disabled = !!TEL.step || vfrRun || S.busy || !S.info;
  an.classList.toggle("primary", !done && !TEL.step && !vfrRun);
}
async function telAnalyze(inAll) {
  if (!S.info || (S.busy && !inAll) || TEL.step) return;
  const key = telKey();
  try {
    TEL.step = "scan"; telButton();
    const sc = await rpc("scan_picture", { state: backendState([S.current]), episode: S.current });
    const scan = sc.progressive ? "progressive" : "interlaced";
    if (S.scan !== scan) changed("scan", scan);
    const how = sc.by === "picture" ? ` (${(100 * sc.combed).toFixed(1)} % of ${sc.frames} frames combed)` : sc.by === "pulldown" ? " (soft telecine flags)" : " (the file's flags)";
    log(`Analyze: ${sc.progressive ? "progressive - whole frames" : "interlaced / telecined"}${how}`, "ok");
    if (sc.progressive) repairFound("frames", "progressive - frame rate kept");
    if (!sc.progressive) {
      TEL.step = "auto"; telButton();
      const quiet = { classList: { add() {}, remove() {} }, textContent: "" };   // Analyze's own button keeps saying "Detecting…"
      await autoGroup("frames", quiet);
    } else status(`Progressive${how}: no field matching, deinterlacing or VFR analysis - the frame rate is kept.`, "ok");
    TEL.done[key] = true;
  } catch { /* logged */ }
  // 3. the VFR analysis (the whole episode, once; the export reuses it). Started BEFORE the button is released, so Analyze
  // keeps saying "VFR pass N %" instead of falling back to "Analyze" and looking as if it ran a second time
  if (TEL.done[key] && S.mode === "vfr" && !S.info.pal && !isProgressive()) vfrStart();
  TEL.step = null; telButton();
}
function vfrStart() {
  if (!S.info) return;
  S.vfrAuto = { state: "running", progress: 0 }; vfrButton();
  rpc("vfr_auto", { state: backendState([S.current]), episode: S.current, start: true })
    .then((a) => { S.vfrAuto = a; vfrButton(); }).catch(() => { S.vfrAuto = { state: "failed" }; vfrButton(); });
}
function autoVfr(r) {
  if (!S.info || S.info.pal || S.mode !== "vfr" || isProgressive()) {
    S.vfrAuto = null; vfrButton();
    if (S.info) rpc("vfr_auto", { state: backendState([S.current]), episode: S.current }).catch(() => {});   // stops a running one
    return;
  }
  if (r.vfr) { S.vfrAuto = { state: "ready" }; vfrButton(); return; }
  rpc("vfr_auto", { state: backendState([S.current]), episode: S.current, start: false })   // how it stands; it runs on Analyze
    .then((a) => { if (!(S.vfrAuto && S.vfrAuto.state === "running" && a.state === "running" && !a.progress)) S.vfrAuto = a; vfrButton(); }).catch(() => {});
}
async function onVfrAuto(m) {
  if (m.episode !== undefined && m.episode !== S.current) return;     // an episode no longer shown
  if (S.mode !== "vfr" || isProgressive() || (S.info && S.info.pal)) return;   // switched away from VFR meanwhile
  if (S.vfrPrev) return;
  if (m.state === "running") { S.vfrAuto = { state: "running", progress: m.progress }; vfrButton(); return; }
  S.vfrAuto = { state: m.state }; vfrButton();
  if (m.state !== "ready" || m.episode !== S.current) return;
  log(`Episode ${S.current + 1}: VFR analysis done - the preview now shows the export's frames and frame rates.`, "ok");
  togglePlay(false);
  if (S.src != null) {
    try { const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: S.src }); S.frame = r.frame; S.total = Math.max(S.total, r.frame + 1); } catch { /* logged */ }
  }
  schedulePreview();
}
function showFrameInfo(n, total, w, h) {
  setTimeout(renderTimeline, 0);
  if (S.info) $("tpTime").textContent = tc(PLAY.on || S.src == null ? Math.round(headSrcFrame()) : S.src);
  const box = $("seekFrame");
  if (document.activeElement !== box) box.value = n ?? "";
  $("frameInfo").textContent = `/ ${Math.max(0, (total || 1) - 1)}` + (w ? ` · ${w}x${h}` : "") + (S.episodes.length > 1 ? ` · ep ${S.current + 1}` : "");
}
function togglePlay(force) {
  const on = force === undefined ? !PLAY.on : force;
  if (on === PLAY.on) return;
  PLAY.on = on;
  $("btnPlay").textContent = PLAY.on ? "❚❚" : "▶";
  $("btnPlay").classList.toggle("on", PLAY.on);
  if (!PLAY.on) { $("playFps").textContent = ""; window.api.rpc("stop_play", {}); setTimeout(requestScopes, 150); return; }   // exact scope of the frame paused on
  if (!S.info) { togglePlay(false); return; }
  if (S.total && S.frame >= S.total - 1) S.frame = 0;
  const views = viewsShown();
  $("playFps").textContent = "…";
  rpc("play", { state: backendState([S.current]), episode: S.current, frame: S.frame || 0, views })
    .catch(() => {}).finally(() => { if (PLAY.on) { PLAY.on = false; $("btnPlay").textContent = "▶"; $("btnPlay").classList.remove("on"); } });
}
function onPlayFrame(m) {
  if (!PLAY.on) return;
  if (m.before) $("imgBefore").src = "data:image/jpeg;base64," + m.before;
  if (m.after) $("imgAfter").src = "data:image/jpeg;base64," + m.after;
  const first = !S.img;
  S.img = { w: m.width, h: m.height }; S.frame = m.frame; S.total = m.total;
  $("rngPos").max = Math.max(1, m.total - 1); $("rngPos").value = m.frame;
  showFrameInfo(m.frame, m.total, m.width, m.height);
  $("playFps").textContent = `${m.fps} fps`;
  liveScope(m.after || (S.view === "before" ? m.before : null));
  if (m.src != null) { if (tlxSkipCut(m.src)) return; tlxFollow(m.src); }
  if (m.rate) { S.rate = m.rate; vfrButton(); }
  if (first) layoutImage();
}
function seekTo(v) {
  if (!S.info) return;
  togglePlay(false);
  const n = parseInt(String(v).trim(), 10);
  if (isNaN(n)) { showFrameInfo(S.frame, S.total); return; }
  S.frame = Math.max(0, Math.min(Math.max(0, S.total - 1), n));
  $("rngPos").value = S.frame;
  schedulePreview(30);
}
function srcFrameAt(k) {                     // preview frame -> source frame of episode k
  const [a, b] = S.episodes[k];
  if (!S.total || S.total < 2) return a;
  return Math.round(a + (S.frame / (S.total - 1)) * (b - a));
}
function trim(which) {
  if (!S.info || !S.episodes.length) return;
  togglePlay(false);
  const k = S.current, ep = S.episodes[k];
  const f = S.src != null && S.src >= ep[0] && S.src < ep[1] ? S.src : srcFrameAt(k);    // exact frame shown when known
  if (which === "in") { if (f >= ep[1] - 2) return; ep[0] = f; S.frame = 0; }
  else { if (f <= ep[0] + 2) return; ep[1] = f + 1; S.frame = null; }
  S.total = 0;
  renderEpisodes(); trimInfo(); schedulePreview(); scheduleScript();
  log(`Episode ${k + 1}: ${which === "in" ? "start" : "end"} set to ${tc(which === "in" ? ep[0] : ep[1])}`, "ok");
}
function trimReset() {
  const k = S.current;
  const d = S.detected[k] || [0, S.info.frames];
  S.episodes[k] = d.slice(); S.frame = null; S.total = 0;
  renderEpisodes(); trimInfo(); schedulePreview(); scheduleScript();
}
function trimFrames(which, v) {
  if (!S.info || !S.episodes.length) return;
  togglePlay(false);
  const ep = S.episodes[S.current], txt = String(v).trim();
  const n = txt.includes(":") ? parseTc(txt) : parseInt(txt, 10);           // a time or a source frame number
  if (isNaN(n)) { trimInfo(); return; }
  if (which === "in") ep[0] = Math.max(0, Math.min(n, ep[1] - 2));
  else ep[1] = Math.max(ep[0] + 2, Math.min(n, S.info.frames));
  S.frame = which === "in" ? 0 : null; S.total = 0;
  renderEpisodes(); trimInfo(); schedulePreview(); scheduleScript();
}
// ---------------------------------------------------------------- whole-video timeline
// The bar shows the selected episode only (+ a little on each side, so its ends can still be dragged outwards); it
// keeps its range while a handle is dragged, then fits the new episode.
function headSrcFrame() {                       // source frame shown in the preview
  const [a, b] = S.episodes[S.current] || [0, 0];
  if (!S.total || S.total < 2 || S.frame == null) return a + ($("rngPos").value / $("rngPos").max) * (b - a);
  return a + (S.frame / (S.total - 1)) * (b - a);
}
function tlSeek(f) {
  const [a, b] = S.episodes[S.current];
  const fr = (f - a) / Math.max(1, b - a);
  if (S.total > 1) S.frame = Math.round(Math.max(0, Math.min(1, fr)) * (S.total - 1));
  else { S.frame = null; $("rngPos").max = 1000; $("rngPos").value = Math.round(1000 * Math.max(0, Math.min(1, fr))); }
  renderTimeline(); schedulePreview();
}

function trimFields(a, b, live = false) {        // In / Out as times (frame in the tooltip), Length
  for (const [id, v] of [["trimStart", a], ["trimEnd", b]]) {
    const x = $(id);
    const fr = typeof TLX !== "undefined" && TLX.frames;            // the Timeline's TC / FR switch
    if (document.activeElement !== x) x.value = fr ? String(v) : tc(v);
    x.title = x.title.split(" · ")[0] + (fr ? ` · ${tc(v)}` : ` · source frame ${v}`);
    x.classList.toggle("live", live);
  }
  $("tpLen").textContent = (typeof TLX !== "undefined" && TLX.frames ? `${b - a} fr` : hmsms(span(a, b)).replace(/\.\d+$/, "")) + (live ? " · release" : "");
}
function trimInfo() {
  renderTimeline();
  if (!S.info || !S.episodes.length) { $("tpEp").textContent = ""; $("tpLen").textContent = ""; return; }
  const [a, b] = S.episodes[S.current], d = S.detected[S.current];
  trimFields(a, b);
  const cut = d && (d[0] !== a || d[1] !== b);
  $("btnTrimReset").classList.toggle("hidden", !cut);
  $("tpEp").textContent = S.episodes.length > 1 ? `Episode ${S.current + 1} of ${S.episodes.length}${cut ? " · trimmed" : ""}` : (cut ? "trimmed" : "whole video");
}

async function renderPreview() {
  if (!S.info || !S.episodes.length) return;
  const seq = ++previewSeq;
  const views = viewsShown();
  $("spinner").classList.remove("hidden");
  try {
    const r = await rpc("preview", { state: backendState([S.current]), episode: S.current, pos: previewPos(), views,
                                     frame: S.total && S.frame != null ? S.frame : null });
    if (seq !== previewSeq) return;
    const mime = r.mime || "image/png";
    if (S.view === "ab") {                       // A = the settings now (left), B = the snapshot (right), same frame
      const sn = (S.snaps || []).find((x) => x.id === S.abSnap) || (S.snaps || [])[0];
      if (sn) {
        S.abSnap = sn.id;
        const rb = await rpc("preview", { state: snapState(sn), episode: S.current, pos: previewPos(), views: ["after"], frame: r.frame });
        if (seq !== previewSeq) return;
        $("imgBefore").src = `data:${mime};base64,` + r.after;
        $("imgAfter").src = `data:${mime};base64,` + rb.after;
      }
    } else {
      if (r.before) $("imgBefore").src = `data:${mime};base64,` + r.before;
      if (r.after) $("imgAfter").src = `data:${mime};base64,` + r.after;
    }
    S.img = { w: r.width, h: r.height };
    S.frame = r.frame; S.total = r.total;
    if (r.src != null && r.src !== S.src) { S.src = r.src; layerUi(); }
    autoVfr(r);
    const rng = $("rngPos");
    rng.max = Math.max(1, r.total - 1); rng.value = r.frame;
    showFrameInfo(r.frame, r.total, r.width, r.height);
    requestScopes();
    S.vfrPrev = !!r.vfr; S.rate = r.rate || 0; vfrButton();
    $("imgWrap").classList.remove("hidden");
    layoutImage();
  } catch (e) { if (S.script) scedPreviewError(e); }   /* superseded or logged */
  if (seq === previewSeq) $("spinner").classList.add("hidden");
}
function stepFrame(d) {
  if (!S.total) return;
  S.frame = Math.max(0, Math.min(S.total - 1, (S.frame || 0) + d));
  $("rngPos").value = S.frame;
  showFrameInfo(S.frame, S.total);
  schedulePreview(30);
}
function viewsShown() { return S.view === "split" || S.view === "sbs" ? ["before", "after"] : S.view === "ab" ? ["after"] : [S.view]; }
function setView(v) {
  if (v === "ab" && !(S.snaps || []).length) { status("Take a snapshot first (＋ Snapshot): A / B compares your settings now with it.", "err"); v = "sbs"; }
  $("abPick").classList.toggle("hidden", v !== "ab");
  const was = PLAY.on; togglePlay(false); S.view = v; setSegView(); if (S.img) schedulePreview(30); if (was) setTimeout(() => togglePlay(true), 300); }
function setSegView() { for (const b of $("segView").children) b.classList.toggle("on", b.dataset.v === S.view); }

// Colour step: the viewer is as wide as the picture (no empty bars beside it), in the middle; the width left over
// goes to the Auto correction (left) and Scopes (right) columns
function fitStageRow() {
  const stage = $("stage"), side = $("gradeSide");
  if (!side) return;
  if (true) {                                     // the scopes moved into the Colour Inspector: the picture keeps the whole viewer
    stage.style.flex = ""; stage.style.width = ""; side.classList.remove("wide"); return;
  }
  const row = stage.parentNode, gap = 8, minSide = 250;    // Auto correction left and Scopes right of the picture
  const n = S.view === "sbs" || S.view === "ab" ? 2 : 1;    // side by side / A-B: two pictures and the gap between them
  const want = Math.round((stage.clientHeight - 4) * S.img.w / S.img.h) * n + (n - 1) * SBS_GAP + 6;
  stage.style.flex = "none";
  const cols = 1;                                           // only the scopes beside the picture (controls: Inspector)
  stage.style.width = Math.max(240, Math.min(row.clientWidth - cols * (minSide + gap), want)) + "px";
}
const SBS_GAP = 6;
function layoutImage() {
  if (!S.img) return;

  fitStageRow();
  const stage = $("stage"), wrap = $("imgWrap"), before = $("imgBefore"), after = $("imgAfter");
  const { w, h } = S.img;
  const sbs = S.view === "sbs" || S.view === "ab", n = sbs ? 2 : 1, gap = sbs ? SBS_GAP : 0;
  const sn = (S.snaps || []).find((x) => x.id === S.abSnap);
  wrap.querySelector(".sbs-b").textContent = S.view === "ab" ? "A · now" : "Before";
  wrap.querySelector(".sbs-a").textContent = S.view === "ab" ? `B · ${sn ? sn.name : "snapshot"}` : "After";
  const scale = S.zoom === "Fit" ? Math.min((stage.clientWidth - 4 - gap) / (w * n), (stage.clientHeight - 4) / h)
                                 : S.zoom === "1:1" ? 1 : parseInt(S.zoom, 10);
  const W = Math.round(w * scale), H = Math.round(h * scale), WW = W * n + gap;
  wrap.style.width = WW + "px"; wrap.style.height = H + "px";
  wrap.classList.toggle("sbs", sbs);
  for (const t of wrap.querySelectorAll(".sbs-tag")) t.classList.toggle("hidden", !sbs);
  wrap.querySelector(".sbs-a").style.left = W + gap + 8 + "px";
  wrap.style.margin = `${Math.max(0, (stage.clientHeight - H) / 2)}px ${Math.max(0, (stage.clientWidth - WW) / 2)}px`;
  stage.classList.toggle("pixel", scale >= 1.5 || S.zoom !== "Fit");
  stage.classList.toggle("zoomed", S.zoom !== "Fit");
  for (const im of [before, after]) { im.style.width = W + "px"; im.style.height = H + "px"; }
  const split = S.view === "split";
  before.classList.toggle("hidden", S.view === "after");
  before.style.marginRight = sbs ? gap + "px" : "";
  after.classList.toggle("hidden", S.view === "before");
  after.classList.toggle("layer", split);
  after.style.clipPath = split ? `inset(0 0 0 ${S.split}%)` : "none";
  $("splitBar").classList.toggle("hidden", !split);
  $("splitBar").style.left = `calc(${S.split}% - 1px)`;
}
function initViewer() {
  const bar = $("splitBar");
  let drag = false;
  bar.onmousedown = (e) => { drag = true; e.preventDefault(); e.stopPropagation(); };
  window.addEventListener("mouseup", () => { drag = false; });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const r = $("imgWrap").getBoundingClientRect();
    S.split = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
    layoutImage();
  });
  $("imgWrap").addEventListener("click", (e) => {             // click = zoom 2x there / back to fit
    if (e.target === bar || drag) return;
    const r = $("imgWrap").getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    S.zoom = S.zoom === "Fit" ? "2x" : "Fit";
    $("selZoom").value = S.zoom; layoutImage();
    if (S.zoom !== "Fit") {
      const st = $("stage"), w = $("imgWrap");
      st.scrollLeft = fx * w.clientWidth - st.clientWidth / 2;
      st.scrollTop = fy * w.clientHeight - st.clientHeight / 2;
    }
  });
  window.addEventListener("keydown", (e) => {
    const typing = /TEXTAREA/.test(document.activeElement.tagName) || (document.activeElement.tagName === "INPUT" && !/checkbox|range/.test(document.activeElement.type));
    if ((e.ctrlKey || e.metaKey) && !typing && !e.altKey) {          // the text boxes keep their own undo
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) { e.preventDefault(); histGo(-1); return; }
      if (k === "y" || (k === "z" && e.shiftKey)) { e.preventDefault(); histGo(1); return; }
      if (k === "s") { e.preventDefault(); saveWorkflow(); return; }
    }
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); stepFrame((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 24 : 1)); }
    else if (e.key === "b" || e.key === "B") setView("before");
    else if (e.key === "a" || e.key === "A") setView("after");
    else if (e.key === "s" || e.key === "S") setView("split");
    else if ((e.key === "d" || e.key === "D") && document.body.classList.contains("step-colour")) setView(S.view === "sbs" ? "split" : "sbs");
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "i" || e.key === "I") trim("in");
    else if (e.key === "o" || e.key === "O") trim("out");
  });
}

// ---------------------------------------------------------------- export
// ---------------------------------------------------------------- batch queue (Deliver / File > Add ticked episodes)
// Every job keeps a frozen copy of its episode's settings: you can keep editing or open other sources meanwhile.
const QUEUE = { jobs: [], running: false, paused: false, cur: null, seq: 0 };
function encLabel() {
  if (S.info && S.info.image) return "PNG";
  const o = (S.opts && S.opts.encoders || []).find((e) => e[0] === S.encoder);
  return (o ? o[1].split(" (")[0].split(" - ")[0] : S.encoder) + (S.enc_rate === "bitrate" ? ` ${S.enc_bitrate} kbit/s` : S.encoder === "ffv1" || S.encoder === "utvideo" ? "" : ` CRF ${S.crf}`);
}
function queueAdd() {
  if (!S.info) return;
  const sel = [...S.checked].sort((x, y) => x - y);
  if (!sel.length) { status("Tick the episodes to export in the Media panel first.", "err"); showLeft("media"); return; }
  saveSettings(); histCommit();
  for (const k of sel) {
    const st = backendState([k]);
    st.per_episode = { [k]: pictureState(k) };
    const [a, b] = S.episodes[k] || [0, 0];
    QUEUE.jobs.push({ id: ++QUEUE.seq, name: basename(S.info.path), ep: k, eps: S.episodes.length, image: !!S.info.image,
                      span: S.info.image ? "" : `${tc(a).slice(0, -4)} – ${tc(b).slice(0, -4)}`, state: st, status: "waiting",
                      prog: 0, eta: "", out: encLabel(), msg: "" });
  }
  log(`Added ${sel.length} job${sel.length > 1 ? "s" : ""} to the queue (${basename(S.info.path)})`, "ok");
  status(`${sel.length} job${sel.length > 1 ? "s" : ""} added - the queue runs in the background while you keep working.`, "ok");
  showDock("queue"); renderQueue();
  if (!QUEUE.running && !QUEUE.paused) queueRun();
}
async function queueRun() {
  if (QUEUE.running) return;
  QUEUE.running = true; QUEUE.paused = false; renderQueue();
  for (;;) {
    if (QUEUE.paused) break;
    const job = QUEUE.jobs.find((j) => j.status === "waiting");
    if (!job) break;
    QUEUE.cur = job; job.status = "running"; job.prog = 0; job.t0 = performance.now(); renderQueue();
    try {
      const outs = await rpc("process", { state: job.state });
      job.status = "done"; job.prog = 1; job.msg = (outs && outs[0]) ? basename(String(outs[0])) : "";
    } catch (e) {
      job.status = QUEUE.stopping ? "stopped" : "failed"; job.msg = QUEUE.stopping ? "stopped" : String(e && e.message || e || "").slice(0, 140);
    }
    QUEUE.stopping = false; QUEUE.cur = null; renderQueue();
  }
  QUEUE.running = false; renderQueue();
  const done = QUEUE.jobs.filter((j) => j.status === "done").length;
  if (!QUEUE.paused && QUEUE.jobs.length) status(`Queue finished: ${done} of ${QUEUE.jobs.length} done.`, done === QUEUE.jobs.length ? "ok" : "err");
}
function queueProgress(cur, tot) {
  const j = QUEUE.cur;
  if (!j || !tot) return;
  j.prog = Math.min(1, cur / tot);
  const el_ = (performance.now() - j.t0) / 1000;
  if (j.prog > 0.01 && el_ > 5) { const left = el_ / j.prog * (1 - j.prog); j.eta = left >= 3600 ? `${Math.floor(left / 3600)} h ${Math.round(left % 3600 / 60)} min` : left >= 60 ? `${Math.round(left / 60)} min` : `${Math.round(left)} s`; }
  const row = document.querySelector(`#queue tr[data-id="${j.id}"]`);
  if (row) { row.querySelector(".prog i").style.width = (100 * j.prog).toFixed(1) + "%"; row.querySelector(".pct").textContent = `${(100 * j.prog).toFixed(0)} %`; row.querySelector(".eta").textContent = j.eta; }
}
function renderQueue() {
  const n = QUEUE.jobs.filter((j) => j.status === "waiting" || j.status === "running").length;
  $("queueCount").textContent = String(n);
  $("btnQueue").classList.toggle("running", QUEUE.running);
  const q = $("queue");
  if (!q) return;
  q.innerHTML = "";
  const bar = el("div", "qbar");
  const go = el("button", QUEUE.running ? "" : "primary", QUEUE.running ? (QUEUE.paused ? "Pausing after this job…" : "Pause after this job") : "▶ Start queue");
  go.disabled = QUEUE.running ? QUEUE.paused : !QUEUE.jobs.some((j) => j.status === "waiting");
  go.onclick = () => { if (QUEUE.running) { QUEUE.paused = true; renderQueue(); } else queueRun(); };
  const clr = el("button", "ghost", "Clear finished");
  clr.disabled = !QUEUE.jobs.some((j) => ["done", "failed", "stopped"].includes(j.status));
  clr.onclick = () => { QUEUE.jobs = QUEUE.jobs.filter((j) => !["done", "failed", "stopped"].includes(j.status)); renderQueue(); };
  const left = QUEUE.jobs.filter((j) => j.status === "waiting").length;
  bar.append(go, el("span", "muted", QUEUE.jobs.length ? `${QUEUE.jobs.length} job(s) · ${left} waiting · runs in the background while you keep working`
                                                            : "Empty. Tick episodes in Media, then Deliver > Add to queue (any number of sources)."), el("span", "push"), clr);
  q.appendChild(bar);
  if (!QUEUE.jobs.length) return;
  const tb = el("table", "qtable");
  tb.innerHTML = "<thead><tr><th>#</th><th>Source</th><th>Part</th><th>Output</th><th>Progress</th><th>ETA</th><th></th></tr></thead>";
  const body = el("tbody");
  QUEUE.jobs.forEach((j, i) => {
    const tr = el("tr", j.status); tr.dataset.id = j.id;
    const part = j.image ? "image" : j.eps > 1 ? `Episode ${j.ep + 1} · ${j.span}` : j.span;
    const st = { waiting: "waiting", running: "", done: "done ✓", failed: "failed", stopped: "stopped" }[j.status];
    const prog = el("td"); prog.innerHTML = `<span class="prog"><i style="width:${(100 * j.prog).toFixed(1)}%"></i></span><span class="pct">${j.status === "running" ? (100 * j.prog).toFixed(0) + " %" : st}</span>`;
    if (j.msg && j.status !== "running") prog.title = j.msg;
    const act = el("td", "qact");
    if (j.status === "waiting") {
      const up = el("button", "ghost", "↑"); up.title = "Earlier"; up.disabled = i === 0;
      up.onclick = () => { [QUEUE.jobs[i - 1], QUEUE.jobs[i]] = [QUEUE.jobs[i], QUEUE.jobs[i - 1]]; renderQueue(); };
      const dn = el("button", "ghost", "↓"); dn.title = "Later"; dn.disabled = i === QUEUE.jobs.length - 1;
      dn.onclick = () => { [QUEUE.jobs[i + 1], QUEUE.jobs[i]] = [QUEUE.jobs[i], QUEUE.jobs[i + 1]]; renderQueue(); };
      const rm = el("button", "ghost", "✕"); rm.title = "Remove from the queue";
      rm.onclick = () => { QUEUE.jobs.splice(i, 1); renderQueue(); };
      act.append(up, dn, rm);
    } else if (j.status === "running") {
      const stop = el("button", "ghost", "Stop"); stop.title = "Stop this job (the queue pauses)";
      stop.onclick = () => { QUEUE.stopping = true; QUEUE.paused = true; rpc("cancel", {}).catch(() => {}); renderQueue(); };
      act.append(stop);
    } else if (j.status === "failed" || j.status === "stopped") {
      const again = el("button", "ghost", "Retry"); again.onclick = () => { j.status = "waiting"; j.prog = 0; j.eta = ""; renderQueue(); if (!QUEUE.running) queueRun(); };
      act.append(again);
    }
    tr.append(el("td", "mono", String(i + 1)), el("td", null, j.name), el("td", null, part), el("td", "mono", j.out), prog, el("td", "mono eta", j.status === "running" ? j.eta : ""), act);
    body.appendChild(tr);
  });
  tb.appendChild(body);
  q.appendChild(tb);
}
// ---------------------------------------------------------------- backend events
function onEvent(m) {
  if (m.event === "log") {
    const cls = /ERROR|WARNING|failed/.test(m.msg) ? "err" : (/OK|Done:|All done|done\.|Imported/.test(m.msg) ? "ok" : "");
    log(m.msg, cls); status(m.msg, cls);
    const q = m.msg.match(/^=== .*\((\d+)\/(\d+)\) ===$/);
    if (q && S.exporting) {
      const n = parseInt(q[1], 10) - 1;
      S.exporting.forEach((k, i) => { if (i < n) S.queue[k] = "done"; else if (i === n) S.queue[k] = "running"; });
      renderQueue();
    }
  } else if (m.event === "play_frame") {
    onPlayFrame(m);
  } else if (m.event === "vfr_auto") {
    onVfrAuto(m);
  } else if (m.event === "progress") {
    $("progBar").style.width = m.tot ? `${(100 * m.cur / m.tot).toFixed(1)}%` : "0";
    progressText(m.cur, m.tot);
    queueProgress(m.cur, m.tot);
  } else if (m.event === "busy") {
    S.busy = m.busy; updateButtons();
    if (!m.busy) { $("progBar").style.width = "0"; progressText(0, 0); }
  } else if (m.event === "engine_build") {
    if (typeof pickEngineProgress === "function") pickEngineProgress(m);
  } else if (m.event === "stderr") {
    if (!/Warning|DeprecationWarning/.test(m.msg)) log(m.msg, "dim");
  } else if (m.event === "exit") {
    if (typeof engineStopped === "function") engineStopped(m.code);
    else { log("Backend stopped - restart the app.", "err"); status("Backend stopped - restart the app.", "err"); }
  }
}

// percentage, speed and time left of the running job (export, VFR analysis ...), from its progress events
const PROG = { tot: 0, t0: 0, c0: 0, rate: 0, last: 0 };
function progressText(cur, tot) {
  const box = $("progText");
  if (!tot) { box.textContent = ""; PROG.tot = 0; return; }
  const now = performance.now();
  if (tot !== PROG.tot || cur < PROG.last) Object.assign(PROG, { tot, t0: now, c0: cur, rate: 0 });   // a new pass / job
  PROG.last = cur;
  const dt = (now - PROG.t0) / 1000;
  if (dt > 1.5 && cur > PROG.c0) {
    const r = (cur - PROG.c0) / dt;
    PROG.rate = PROG.rate ? PROG.rate * 0.8 + r * 0.2 : r;            // smoothed: the ETA does not jump around
    PROG.t0 = now; PROG.c0 = cur;
  }
  const pct = Math.min(100, 100 * cur / tot);
  let txt = `${pct.toFixed(1)} %`;
  if (PROG.rate > 0) {
    const left = Math.max(0, (tot - cur) / PROG.rate);
    const h = Math.floor(left / 3600), mn = Math.floor(left % 3600 / 60), sc = Math.floor(left % 60);
    txt += ` · ${PROG.rate >= 10 ? Math.round(PROG.rate) : PROG.rate.toFixed(1)} fps · ETA ${h ? h + ":" + String(mn).padStart(2, "0") : mn}:${String(sc).padStart(2, "0")}`;
  }
  box.textContent = txt;
}

// ---------------------------------------------------------------- wiring
function bind() {
  $("btnAddFilterTop").onclick = () => $("vsfNewGroup").click();   // the top bar's + Add a filter (the bottom one is hidden)
  $("btnAddModelTop").onclick = () => {                           // + Add a model: the library with your AI models only
    vsfMigrate();
    const ng = vsfNewGroup("colour");
    S.vs_groups.push(ng); vsgChanged();
    vsfBrowser(S.vs_groups.find((x) => x.id === ng.id) || ng, null, { fresh: true, only: "ai" });
  };
  $("stepsDeleteAll").onclick = () => { changed("removed_steps", [...STATIC_IDS]); log("All built-in steps deleted - add your own with + Add a filter", "ok"); };
  $("stepsRestoreAll").onclick = stepsRestoreAll;
  $("vsfNewGroup").onclick = () => {               // straight to the library; the group stays only if something is added
    vsfMigrate();
    const ng = vsfNewGroup("colour");
    S.vs_groups.push(ng); vsgChanged();
    const g = S.vs_groups.find((x) => x.id === ng.id) || ng;
    vsfBrowser(g, null, { fresh: true });
  };
  for (const sg of document.querySelectorAll(".rep-grip")) {      // drag a Repair group (job "repair_order")
    sg.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", "rep:" + sg.dataset.step); e.dataTransfer.setData("application/x-eva-step-" + sg.dataset.step, "1"); e.dataTransfer.effectAllowed = "copyMove"; sg.closest(".group").classList.add("dragging"); });
    sg.addEventListener("dragend", () => { sg.closest(".group").classList.remove("dragging"); vsfDropMark(null); });
  }
  // every step (also the ones that cannot be moved) can be dragged by its header onto the Timeline
  for (const grp of document.querySelectorAll('[data-panel="adjust"] > .group[data-gid]:not(.vsf-group)')) {
    const head = grp.querySelector(":scope > .group-head");
    if (!head) continue;
    head.draggable = true;
    head.addEventListener("dragstart", (e) => {
      if (e.target !== head) return;               // the handle has its own (it also moves the step in the order)
      e.dataTransfer.setData("application/x-eva-step-" + grp.dataset.gid, "1"); e.dataTransfer.effectAllowed = "copy";
      grp.classList.add("dragging");
    });
    head.addEventListener("dragend", () => grp.classList.remove("dragging"));
  }
  initEngineDialog();
  initGradeTabs();
  $("btnUndo").onclick = () => histGo(-1); $("btnRedo").onclick = () => histGo(1);
  $("btnWfSave").onclick = saveWorkflow; $("btnWfLoad").onclick = loadWorkflow;
  histButtons();
  for (const b of $$("#stepNav button")) b.onclick = () => goto(b.dataset.step);
  for (const b of $$("[data-goto]")) b.onclick = () => goto(b.dataset.goto);
  const pickFile = async () => openPath(await window.api.openVideo());
  const pickFolder = async () => openPath(await window.api.openFolder("Choose a DVD folder (VIDEO_TS or the folder that contains it)"));
  $("btnOpenFile").onclick = pickFile;
  $("btnOpenFolder").onclick = pickFolder;
  $("btnChange").onclick = () => { $("dropZone").classList.remove("hidden", "compact"); $("dropZone").classList.add("compact"); };
  $("btnImport").onclick = importTitle;
  $("btnCancelDisc").onclick = () => { $("discBox").classList.add("hidden"); if (!S.info) $("dropZone").classList.remove("compact"); };

  // drag & drop anywhere
  let depth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; document.body.classList.add("dragging"); });
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; document.body.classList.remove("dragging"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; document.body.classList.remove("dragging");
    const f = e.dataTransfer.files[0];
    if (f && !S.busy) { goto("source"); openPath(window.api.pathForFile(f)); }
  });

  $("chkAllEp").onchange = () => { S.checked = $("chkAllEp").checked ? new Set(S.episodes.map((_, k) => k)) : new Set(); renderEpisodes(); };
  // new episode list: the current settings go to the first episode, the others start from the defaults
  $("btnChapters").onclick = () => { S.episodes = S.detected.map((e) => e.slice()); S.epSet = S.episodes.map(() => null); S.checked = new Set(S.episodes.map((_, k) => k)); S.current = 0; renderEpisodes(); schedulePreview(); };
  $("btnWhole").onclick = () => { S.episodes = [[0, S.info.frames]]; S.epSet = [null]; S.checked = new Set([0]); S.current = 0; renderEpisodes(); schedulePreview(); };
  $("btnCopyAll").onclick = copySettingsToAll;
  $("btnAddEp").onclick = () => {
    const [a, b] = S.episodes[S.current] || [0, S.info.frames];
    epSetAlign();
    S.epSet.splice(S.current + 1, 0, pictureSnap());            // the duplicate starts with the same settings
    S.episodes.splice(S.current + 1, 0, [a, b]);
    S.checked = new Set([...S.checked].map((k) => (k > S.current ? k + 1 : k)).concat([S.current + 1]));
    renderEpisodes();
  };
  $("btnDelEp").onclick = () => {
    if (S.episodes.length <= 1) { status("At least one episode is needed - use One episode to start over.", "err"); return; }
    epSetAlign();
    S.episodes.splice(S.current, 1); S.epSet.splice(S.current, 1);
    S.checked = new Set([...S.checked].filter((k) => k !== S.current).map((k) => (k > S.current ? k - 1 : k)));
    S.current = Math.max(0, Math.min(S.current, S.episodes.length - 1));
    pictureLoad(S.epSet[S.current] || pictureFresh());            // the episode now shown
    renderEpisodes(); sync(); schedulePreview();
  };

  $("btnAnalyze").onclick = analyze;
  for (const b of $$(".auto-btn")) b.onclick = async () => {
    // FieldNet: the Composite Auto (next to its dials) measures blocking and halos and sets the MPEG / Halos dials
    if (b.dataset.auto === "composite" && (S.ai_model === "field" || S.ai_model === "uni")) { await autoGroup("block", b); await autoGroup("ghost", b); return; }
    autoGroup(b.dataset.auto, b);
  };
  $("btnAdvanced").onclick = () => {
    const box = $("advBox"); box.classList.toggle("hidden");
    $("btnAdvanced").textContent = box.classList.contains("hidden") ? "Fine-tune ▾" : "Fine-tune ▴";
  };
  $("btnOutDir").onclick = async () => { const p = await window.api.openFolder("Output folder"); if (p) changed("out_dir", p); };
  $("btnShowOut").onclick = () => S.out_dir && window.api.showPath(S.out_dir);
  $("btnExport").onclick = queueAdd;
  $("btnGradeReset").onclick = () => {
    const ly = S.layers[S.layerSel];
    if (ly) { ly.grade = structuredClone(GRADE0); sync(); schedulePreview(); scheduleScript(); return; }   // layer: no changes
    S.grade = structuredClone(GRADE0);
    for (const k of Object.keys(S.auto)) if (k.startsWith("grade.")) delete S.auto[k];
    for (const id of Object.keys(S.autoUse)) S.autoUse[id] = false;          // measurement kept: tick a line to use it again
    sync(); syncColourUi(); schedulePreview(); scheduleScript();
  };
  $("btnAutoColour").onclick = () => autoColour(false);
  $("btnAutoWB").onclick = () => autoColour(true);
  $("btnSettings").onclick = openSettings;
  $("btnGithub").onclick = () => window.api.openGithub();
  $("btnDeps").onclick = () => window.api.depsOpen();
  $("encPreset").onchange = () => changed("enc_preset", $("encPreset").value);
  $("encTune").onchange = () => changed("enc_tune", $("encTune").value);
  $("btnSettingsClose").onclick = () => $("settingsDlg").classList.add("hidden");
  $("settingsDlg").onclick = (e) => { if (e.target === $("settingsDlg")) $("settingsDlg").classList.add("hidden"); };
  const box = $("scriptBox");
  box.addEventListener("input", () => { S.script = box.value; scriptStateUi(); schedulePreview(500); });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.defaultPrevented) { e.preventDefault(); const a = box.selectionStart; box.setRangeText("    ", a, box.selectionEnd, "end"); box.dispatchEvent(new Event("input")); }
  });
  $("btnScriptReset").onclick = () => { S.script = null; S.scriptFile = null; refreshScript(true); schedulePreview(); sync(); };
  $("btnScriptCheck").onclick = checkScript;
  $("btnScriptSave").onclick = saveScript;
  $("btnScriptOpen").onclick = () => scriptOpen();
  $("btnAutoCrop").onclick = autoCrop;
  $("btnReport").onclick = collectReport;
  $("btnVsview").onclick = () => { S.vsviewLive = true; rpc("vsview", { state: vsviewState() }).catch(() => {}); };
  $("btnStop").onclick = () => rpc("cancel", {}).catch(() => {});
  $("btnLog").onclick = () => showDock("log");

  for (const b of $("segView").children) b.onclick = () => setView(b.dataset.v);
  $("btnPlay").onclick = () => togglePlay();
  $("btnVfrPrev").onclick = vfrStart;
  $("btnVfrAnalyze").onclick = telAnalyze;
  $("btnTrimIn").onclick = () => trim("in");
  $("btnTrimOut").onclick = () => trim("out");
  $("btnTrimReset").onclick = trimReset;
  $("seekFrame").onkeydown = (e) => { if (e.key === "Enter") { seekTo($("seekFrame").value); $("seekFrame").blur(); } else if (e.key === "Escape") { showFrameInfo(S.frame, S.total); $("seekFrame").blur(); } };
  $("seekFrame").onchange = () => seekTo($("seekFrame").value);
  for (const [id, w] of [["trimStart", "in"], ["trimEnd", "out"]]) {
    $(id).onkeydown = (e) => { if (e.key === "Enter") { trimFrames(w, $(id).value); $(id).blur(); } };
    $(id).onchange = () => trimFrames(w, $(id).value);
  }
  $("selZoom").onchange = () => { S.zoom = $("selZoom").value; layoutImage(); };
  $("btnPrev").onclick = (e) => stepFrame(e.shiftKey ? -24 : -1);
  $("btnNext").onclick = (e) => stepFrame(e.shiftKey ? 24 : 1);
  $("rngPos").oninput = () => { togglePlay(false); if (S.total) S.frame = parseInt($("rngPos").value, 10); schedulePreview(); };
  window.addEventListener("resize", layoutImage);
  initViewer();
  initTimeline();
  initFilterDrop();
  initRangeSelection();
  window.api.onEvent(onEvent);
}

function buildAdvanced() {
  const box = $("advBox");
  box.innerHTML = "";
  for (const [key, label, def, help] of S.opts.advanced) {
    const f = el("div", "field");
    const l = el("label", null, label.replace(/ \(.*\)$/, "")); l.title = label;
    const inp = el("input"); inp.type = "text"; inp.placeholder = `default ${def}`; inp.value = S.adv[key] || "";
    inp.onchange = () => { S.adv[key] = inp.value.trim(); saveSettings(); };
    f.append(l, inp, el("div", "field-hint", help));
    box.appendChild(f);
  }
}

// ---------------------------------------------------------------- script step
let scriptTimer = null;
function scheduleScript() {
  if (S.script || !(typeof scriptWinOpen === "function" && scriptWinOpen())) return;   // follows the settings while the script window is open and the script not edited
  clearTimeout(scriptTimer); scriptTimer = setTimeout(() => refreshScript(), 300);
}
async function refreshScript(force = false) {
  if (!S.info || (S.script && !force)) { scriptStateUi(); return; }
  try {
    const t = await rpc("script", { state: backendState([S.current]), episode: S.current });
    S.scriptGen = t;
    if (!S.script) { $("scriptBox").value = t; scedRefresh(); }
    liveScriptRender(t);
  } catch { /* logged */ }
  scriptStateUi();
}
function scriptStateUi() {
  const ed = !!S.script;
  const chip = $("scriptState");
  chip.className = "chip " + (ed ? "edited" : "auto");
  const ext = ed && scriptIsExternal(S.script);
  chip.textContent = !ed ? "from settings" : ext ? `your own script${S.scriptFile ? " · " + S.scriptFile : ""} - preview and export run it as written`
                                                 : "edited - preview and export use this text";
  $("scriptBox").classList.toggle("edited", ed);
  $("btnScriptReset").disabled = !ed;
  $("scriptFlag").classList.toggle("hidden", !ed);
  const sb = $("btnScriptSide"); if (sb) { sb.classList.toggle("edited", ed); sb.innerHTML = ed ? "&lt;/&gt; Script · yours" : "&lt;/&gt; Script"; }
  $("exportNote").textContent = !ed ? "" : ext ? "Export runs your own script (Filtering › Script); each episode is its part of the output, with this file's sound and subtitles."
                                                 : "Export uses your edited script (Script step) for every ticked episode.";
  const note = $("scedExt");
  if (note) note.classList.toggle("hidden", !ext);
}
async function checkScript() {
  const msg = $("scriptMsg");
  msg.className = "note"; msg.textContent = "Running the script...";
  try {
    const r = await rpc("check_script", { state: backendState([S.current]), episode: S.current, text: $("scriptBox").value });
    if (r.ok) { msg.className = "note ok"; msg.textContent = `OK: ${r.frames} frames, ${r.width}x${r.height}, ${r.format}, ${r.fps} fps (preview rate).`; scedShowError(null); }
    else { msg.className = "note err"; msg.textContent = `${r.line ? "Line " + r.line + ": " : ""}${r.error}${r.where && !r.line ? "  -  " + r.where : ""}`; scedShowError(r.line, r.error); }
  } catch (e) { msg.className = "note err"; msg.textContent = String(e.message || e); }
}
async function saveScript() {
  const name = (S.info ? basename(S.info.path).replace(/\.[^.]+$/, "") : "script") + `_ep${S.current + 1}.vpy`;
  const p = await window.api.saveText(name, $("scriptBox").value, [{ name: "VapourSynth script", extensions: ["vpy", "py"] }]);
  if (p) { log(`Script saved: ${p}`, "ok"); status(`Script saved: ${p}`, "ok"); }
}

// ---------------------------------------------------------------- export: encoder options per encoder
let ENC_PREV = null;
function encoderUi() {
  const T = S.opts && S.opts.enc_table; if (!T) return;
  const e = T[S.encoder] || T.x264;
  if (ENC_PREV !== S.encoder) {                    // new encoder: its defaults
    if (ENC_PREV !== null) { S.enc_preset = e.preset; S.enc_tune = e.tune; if (e.q) S.crf = e.q[2]; S.enc_depth = e.depth; if (!e.chroma.includes(S.enc_chroma)) S.enc_chroma = e.chroma[0]; }
    ENC_PREV = S.encoder;
    const fill = (sel, items, v) => { sel.innerHTML = ""; for (const it of items) { const o = el("option", null, it); o.value = it; sel.appendChild(o); } sel.value = v; };
    fill($("encPreset"), e.presets, S.enc_preset || e.preset); fill($("encTune"), e.tunes, S.enc_tune || e.tune);
    const seg = (box, items, key, fmt) => { box.innerHTML = ""; for (const v of items) { const b = el("button", null, fmt(v)); b.dataset.v = String(v); b.onclick = () => changed(key, key === "enc_depth" ? Number(v) : String(v)); box.appendChild(b); } };
    seg($("encDepth"), e.depths, "enc_depth", (v) => `${v}-bit`);
    seg($("encChroma"), e.chroma, "enc_chroma", (v) => `${v[0]}:${v[1]}:${v[2]}`);
  }
  if (!e.presets.includes(S.enc_preset)) S.enc_preset = e.preset;
  if (!e.tunes.includes(S.enc_tune)) S.enc_tune = e.tune;
  if (!e.depths.includes(Number(S.enc_depth))) S.enc_depth = e.depth;
  if (!e.chroma.includes(String(S.enc_chroma))) S.enc_chroma = e.chroma[0];
  $("encPreset").value = S.enc_preset; $("encTune").value = S.enc_tune;
  $("encPresetLbl").textContent = S.encoder === "prores" ? "Profile" : "Preset";
  $("encPresetRow").classList.toggle("hidden", !e.presets.length);
  $("encTuneRow").classList.toggle("hidden", !e.tunes.length);
  $("encRateRow").classList.toggle("hidden", !e.bitrate);
  const br = e.bitrate && S.enc_rate === "bitrate";
  $("encQRow").classList.toggle("hidden", !e.q || br);
  $("encBrRow").classList.toggle("hidden", !br);
  if (e.q) $("encQLbl").textContent = `Quality (${e.q[3]})`;
  $("encKeyRow").classList.toggle("hidden", ["ffv1", "utvideo", "prores"].includes(S.encoder));
  for (const b of $("encDepth").children) b.classList.toggle("on", b.dataset.v === String(S.enc_depth));
  for (const b of $("encChroma").children) b.classList.toggle("on", b.dataset.v === String(S.enc_chroma));
  $("encDepthRow").classList.toggle("hidden", e.depths.length < 2);         // one possible depth: nothing to choose
  $("encNote").textContent = {
    x264: "Most compatible. CRF 14-18 is near transparent for DVD material; tune animation for cartoons.",
    x265: "Smaller files than x264 at the same quality, slower. CRF 16-20.",
    svtav1: "Smallest files; preset 4-6 is a good balance (lower = slower, better). CRF 20-30.",
    nvenc_h264: "GPU encoder: very fast, bigger files than x264 at the same quality. 8-bit only.",
    nvenc_hevc: "GPU HEVC: very fast, 10-bit. CQ 18-24.",
    nvenc_av1: "GPU AV1 (RTX 40/50 series): very fast and small. CQ 22-30.",
    prores: "For editing in Premiere / Resolve / Final Cut. Large files. 4444 profiles keep full colour.",
    ffv1: "Mathematically lossless, for archiving or further work. Very large files.",
    utvideo: "Lossless and very fast to decode, 8-bit. Very large files.",
  }[S.encoder] || "";
}

// ---------------------------------------------------------------- settings
async function tempUi() {
  let t; try { t = await rpc("temp_info", { state: { temp_dir: S.temp_dir || "" } }); } catch { return; }
  $("tempDir").value = t.path; $("tempDir").title = t.path;
  $("tempInfo").textContent = t.exists ? `${t.mb >= 1000 ? (t.mb / 1000).toFixed(1) + " GB" : t.mb + " MB"} in use` : "empty";
  $("btnTempDefault").classList.toggle("hidden", !S.temp_dir);
  $("btnTempOpen").disabled = !t.exists; $("btnTempClear").disabled = !t.exists || !t.mb || S.busy;
}
function initTempUi() {
  $("btnTempChange").onclick = async () => { const p = await window.api.openFolder("Folder for temporary files"); if (p) { changed("temp_dir", p); tempUi(); } };
  $("btnTempDefault").onclick = () => { changed("temp_dir", ""); tempUi(); };
  $("btnTempOpen").onclick = () => window.api.showPath($("tempDir").value);
  $("btnTempClear").onclick = async () => {
    if (!confirm(`Delete the temporary files in ${$("tempDir").value}?
VFR analyses are made again when needed; finished videos are not touched.`)) return;
    try { const r = await rpc("temp_clear", { state: { temp_dir: S.temp_dir || "" } }); if (r.error) status(r.error, "err"); else log(`Temporary files deleted (${r.removed} source folder${r.removed === 1 ? "" : "s"}).`, "ok"); } catch { /* logged */ }
    tempUi();
  };
}
async function openSettings() {
  $("settingsDlg").classList.remove("hidden");
  tempUi();
  try { S.env = await rpc("env", {}); } catch { return; }
  const e = S.env, box = $("envInfo");
  box.innerHTML = "";
  const row = (k, v) => box.append(el("span", null, k), el("b", null, v));
  row("GPU", e.cuda ? e.gpu : "no CUDA GPU - models run on the CPU");
  row("PyTorch", e.torch || "not installed");
  row("TensorRT", e.tensorrt ? e.tensorrt_version : "not installed (pip install tensorrt)");
  row("Compiled engines", `${e.engines} (${e.engines_mb} MB in engines\\)`);
  const ok = e.tensorrt && e.cuda;
  $$('[data-bind="backend"] button[data-v="tensorrt"]')[0].disabled = !ok;
  if (!ok && S.backend === "tensorrt") changed("backend", "pytorch");
}

// ---------------------------------------------------------------- Colour step: sliders, wheels, curves, scopes
function buildSliders() {
  for (const sl of $$(".slider[data-bind]")) {
    const key = sl.dataset.bind, min = +sl.dataset.min, max = +sl.dataset.max, step = +sl.dataset.step, def = +sl.dataset.def;
    const r = el("input"); r.type = "range"; r.min = min; r.max = max; r.step = step;
    const t = el("input"); t.type = "text"; t.className = "mono";
    const dec = String(step).includes(".") ? String(step).split(".")[1].length : 0;
    const clamp = (v) => parseFloat(Math.max(min, Math.min(max, Math.round(v / step) * step)).toFixed(4));
    r.oninput = () => { t.value = (+r.value).toFixed(dec); setv(key, clamp(+r.value)); sync(); schedulePreview(); };
    r.onchange = () => changed(key, clamp(+r.value));
    t.onchange = () => { const v = parseFloat(t.value.replace(",", ".")); changed(key, isNaN(v) ? getv(key) : clamp(v)); };
    sl.ondblclick = () => {                      // reset: the focused slider / box would otherwise keep its old number
      if (document.activeElement && sl.contains(document.activeElement)) document.activeElement.blur();
      r.value = def; t.value = (+def).toFixed(dec);
      changed(key, def);
    };
    sl.append(r, t);
    sl.dataset.dec = dec;
  }
}
function syncSliders() {
  for (const sl of $$(".slider[data-bind]")) {
    const v = getv(sl.dataset.bind), [r, t] = sl.querySelectorAll("input");
    if (document.activeElement !== r) r.value = v;
    if (document.activeElement !== t) t.value = (+v).toFixed(+sl.dataset.dec);
    sl.classList.toggle("moved", Math.abs(+v - +sl.dataset.def) > 1e-9);
  }
}

// colour wheels: value = [brightness, Cb, Cr]; dot position = (Cb, Cr) inside the unit circle
function wheelVal(name) { const v = S.grade[name]; return Array.isArray(v) ? v : [0, 0, 0]; }
function buildWheels() {
  for (const w of $$(".wheel[data-wheel]")) {
    const name = w.dataset.wheel, size = 124;
    const cv = el("canvas"); cv.width = size; cv.height = size; cv.className = "wheel-cv";
    const lab = el("div", "wheel-lab", w.dataset.label);
    const lum = el("input"); lum.type = "range"; lum.min = -1; lum.max = 1; lum.step = 0.01; lum.className = "wheel-lum";
    lum.title = "Brightness of this range (double-click = 0)";
    const val = el("div", "wheel-val mono");
    w.append(lab, cv, lum, val);
    const set = (v, commit) => { S.grade[name] = v; if (commit) changed(`grade.${name}`, v); else { drawWheel(w); schedulePreview(); } };
    let drag = false, fine = false, start = null;
    const at = (e) => {
      const r = cv.getBoundingClientRect(), R = size / 2 - 6;
      let u = ((e.clientX - r.left) * size / r.width - size / 2) / R, v = -((e.clientY - r.top) * size / r.height - size / 2) / R;
      if (fine && start) { u = start[1] + (u - start[3]) * 0.2; v = start[2] + (v - start[4]) * 0.2; }
      const m = Math.hypot(u, v); if (m > 1) { u /= m; v /= m; }
      return [wheelVal(name)[0], +u.toFixed(3), +v.toFixed(3)];
    };
    cv.onpointerdown = (e) => {
      drag = true; fine = e.shiftKey; cv.setPointerCapture(e.pointerId);
      if (fine) { const r = cv.getBoundingClientRect(), R = size / 2 - 6; const cur = wheelVal(name); start = [0, cur[1], cur[2], ((e.clientX - r.left) * size / r.width - size / 2) / R, -((e.clientY - r.top) * size / r.height - size / 2) / R]; }
      else { start = null; set(at(e), false); }
    };
    cv.onpointermove = (e) => { if (drag) set(at(e), false); };
    cv.onpointerup = (e) => { if (drag) { drag = false; set(at(e), true); } };
    cv.ondblclick = () => { const v = wheelVal(name); set([v[0], 0, 0], true); };
    lum.oninput = () => { const v = wheelVal(name); set([+lum.value, v[1], v[2]], false); };
    lum.onchange = () => { const v = wheelVal(name); set([+lum.value, v[1], v[2]], true); };
    lum.ondblclick = () => { const v = wheelVal(name); set([0, v[1], v[2]], true); };
    w._bg = wheelBackground(size);
    drawWheel(w);
  }
}
function wheelBackground(size) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  const ctx = c.getContext("2d"), img = ctx.createImageData(size, size), R = size / 2 - 6;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = (x - size / 2) / R, v = -(y - size / 2) / R, m = Math.hypot(u, v), i = 4 * (y * size + x);
    if (m > 1.04) { img.data[i + 3] = 0; continue; }
    const k = Math.min(1, m) * 90;
    img.data[i] = 70 + k * (1.402 * v); img.data[i + 1] = 70 + k * (-0.344 * u - 0.714 * v); img.data[i + 2] = 70 + k * (1.772 * u);
    img.data[i + 3] = m > 1 ? 255 * (1.04 - m) / 0.04 : 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function drawWheel(w) {
  const name = w.dataset.wheel, cv = w.querySelector("canvas"), ctx = cv.getContext("2d"), size = cv.width, R = size / 2 - 6;
  const [l, u, v] = wheelVal(name);
  ctx.clearRect(0, 0, size, size); ctx.drawImage(w._bg, 0, 0);
  ctx.strokeStyle = "#0008"; ctx.beginPath(); ctx.moveTo(size / 2, 6); ctx.lineTo(size / 2, size - 6); ctx.moveTo(6, size / 2); ctx.lineTo(size - 6, size / 2); ctx.stroke();
  const x = size / 2 + u * R, y = size / 2 - v * R;
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 8, 0, 2 * Math.PI); ctx.stroke();
  const lum = w.querySelector(".wheel-lum"); if (document.activeElement !== lum) lum.value = l;
  w.querySelector(".wheel-val").textContent = `${l >= 0 ? "+" : ""}${(+l).toFixed(2)} · ${Math.round(100 * Math.hypot(u, v))}%`;
  w.classList.toggle("moved", Math.abs(l) + Math.abs(u) + Math.abs(v) > 1e-6);
}

// curves: monotone cubic, identical to vsgraph.monotone_curve
function monotone(pts) {
  let p = [...(pts || [])].map(([a, b]) => [+a, +b]).sort((a, b) => a[0] - b[0]);
  if (!p.length || p[0][0] > 0) p.unshift([0, 0]);
  if (p[p.length - 1][0] < 1) p.push([1, 1]);
  p = p.filter((q, i) => i === 0 || q[0] - p[i - 1][0] > 1e-6);
  const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]), n = xs.length;
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); d.push((ys[i + 1] - ys[i]) / h[i]); }
  const m = new Array(n).fill(0); m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i]);
  return (x) => {
    let k = 0; while (k < n - 2 && x > xs[k + 1]) k++;
    const t = (x - xs[k]) / h[k], t2 = t * t, t3 = t2 * t;
    const y = (2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h[k] * m[k] + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h[k] * m[k + 1];
    return Math.max(0, Math.min(1, y));
  };
}
const CURVE = { ch: "m", drag: -1 };
const CURVE_COL = { m: "#e8eaf0", r: "#ff6b6b", g: "#5ddc8a", b: "#6b9bff" };
function curvePts(ch) { const c = (S.grade.curves || {})[ch]; return c && c.length ? c.map((p) => [...p]) : [[0, 0], [1, 1]]; }
function drawCurves() {
  const cv = $("curveCanvas"); if (!cv) return;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, P = 8, sx = (x) => P + x * (W - 2 * P), sy = (y) => H - P - y * (H - 2 * P);
  ctx.fillStyle = "#0b0c10"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#262b36"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(sx(i / 4), sy(0)); ctx.lineTo(sx(i / 4), sy(1)); ctx.moveTo(sx(0), sy(i / 4)); ctx.lineTo(sx(1), sy(i / 4)); ctx.stroke(); }
  ctx.strokeStyle = "#353b48"; ctx.beginPath(); ctx.moveTo(sx(0), sy(0)); ctx.lineTo(sx(1), sy(1)); ctx.stroke();
  for (const ch of ["m", "r", "g", "b"]) {
    const f = monotone(curvePts(ch)), act = ch === CURVE.ch;
    if (!act && curvePts(ch).length <= 2 && curvePts(ch).every(([x, y]) => Math.abs(x - y) < 1e-6)) continue;
    ctx.strokeStyle = CURVE_COL[ch]; ctx.globalAlpha = act ? 1 : 0.35; ctx.lineWidth = act ? 2 : 1;
    ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = i / 200; i ? ctx.lineTo(sx(x), sy(f(x))) : ctx.moveTo(sx(x), sy(f(x))); } ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const [x, y] of curvePts(CURVE.ch)) { ctx.fillStyle = CURVE_COL[CURVE.ch]; ctx.beginPath(); ctx.arc(sx(x), sy(y), 4.5, 0, 2 * Math.PI); ctx.fill(); }
  for (const b of $("curveSeg").children) b.classList.toggle("on", b.dataset.v === CURVE.ch);
}
function initCurves() {
  const cv = $("curveCanvas"), P = 8;
  const pos = (e) => { const r = cv.getBoundingClientRect(), W = r.width, H = r.height; return [Math.max(0, Math.min(1, (e.clientX - r.left - P * W / cv.width) / (W - 2 * P * W / cv.width))), Math.max(0, Math.min(1, 1 - (e.clientY - r.top - P * H / cv.height) / (H - 2 * P * H / cv.height)))]; };
  const near = (pts, [x, y]) => pts.findIndex(([a, b]) => Math.hypot(a - x, b - y) < 0.035);
  const save = (pts, commit) => {
    const c = { ...(S.grade.curves || {}) }; c[CURVE.ch] = pts; S.grade.curves = c;
    drawCurves(); if (commit) changed("grade.curves", c); else schedulePreview();
  };
  cv.onpointerdown = (e) => {
    const p = pos(e), pts = curvePts(CURVE.ch); let k = near(pts, p);
    if (k < 0) { pts.push([p[0], monotone(pts)(p[0])]); pts.sort((a, b) => a[0] - b[0]); k = near(pts, [p[0], monotone(pts)(p[0])]); save(pts, false); }
    CURVE.drag = k; cv.setPointerCapture(e.pointerId);
  };
  cv.onpointermove = (e) => {
    if (CURVE.drag < 0) return;
    const p = pos(e), pts = curvePts(CURVE.ch), k = CURVE.drag;
    if (k === 0) p[0] = 0; else if (k === pts.length - 1) p[0] = 1;
    else p[0] = Math.max(pts[k - 1][0] + 0.01, Math.min(pts[k + 1][0] - 0.01, p[0]));
    pts[k] = [+p[0].toFixed(4), +p[1].toFixed(4)]; save(pts, false);
  };
  cv.onpointerup = () => { if (CURVE.drag >= 0) { CURVE.drag = -1; save(curvePts(CURVE.ch), true); } };
  cv.ondblclick = (e) => { const pts = curvePts(CURVE.ch), k = near(pts, pos(e)); if (k > 0 && k < pts.length - 1) { pts.splice(k, 1); save(pts, true); } };
  for (const b of $("curveSeg").children) b.onclick = () => { CURVE.ch = b.dataset.v; drawCurves(); };
  $("btnCurveReset").onclick = () => save([[0, 0], [1, 1]], true);
  drawCurves();
}

// scopes of the frame shown
const SCOPE = { kind: "waveform", timer: null, seq: 0 };
function requestScopes() {
  if (!scopesShown() || !S.info) return;
  clearTimeout(SCOPE.timer);
  SCOPE.timer = setTimeout(async () => {
    const seq = ++SCOPE.seq;
    $("scopeWait").textContent = "…"; $("scopeWait").classList.remove("hidden");
    try {
      const r = await rpc("scopes", { state: backendState([S.current]), episode: S.current, frame: S.frame || 0, kind: SCOPE.kind });
      if (seq !== SCOPE.seq) return;
      $("scopeImg").src = "data:image/png;base64," + r.png; $("scopeWait").classList.add("hidden");
      $("scopeNote").textContent = { waveform: "Brightness of every column, bottom = black. The two orange lines are TV black (16) and white (235): keep the picture between them.",
        parade: "Waveform of red, green and blue side by side: neutral greys line up at the same height in all three.",
        vectorscope: "Colour direction and strength, centre = no colour. Squares = 75 % colour bar targets; the orange line is where skin tones fall.",
        histogram: "How many pixels have each brightness, per channel; the orange lines are TV black and white." }[SCOPE.kind];
    } catch { $("scopeWait").textContent = "-"; }
  }, 250);
}
// live scopes while playing: every frame Play sends to the viewer is also drawn as a scope here (the same four as
// scopes.py, from the shown JPEG at ~1/2 size), about 10 times a second; on pause the exact one comes back
const LIVE = { busy: false, last: 0, img: new Image(), cv: document.createElement("canvas"), out: document.createElement("canvas") };
function liveScope(b64) {
  if (!scopesShown() || LIVE.busy || !b64) return;
  const now = performance.now();
  if (now - LIVE.last < 95) return;
  LIVE.busy = true; LIVE.last = now;
  LIVE.img.onload = () => {
    try { drawLiveScope(LIVE.img); } catch { /* a frame skipped */ }
    LIVE.busy = false;
  };
  LIVE.img.onerror = () => { LIVE.busy = false; };
  LIVE.img.src = "data:image/jpeg;base64," + b64;
}
function drawLiveScope(img) {
  const sw = Math.min(360, img.naturalWidth), sh = Math.max(1, Math.round(img.naturalHeight * sw / img.naturalWidth));
  const cv = LIVE.cv; cv.width = sw; cv.height = sh;
  const cx = cv.getContext("2d", { willReadFrequently: true }); cx.drawImage(img, 0, 0, sw, sh);
  const px = cx.getImageData(0, 0, sw, sh).data, kind = SCOPE.kind;
  const BG = [14, 16, 20];
  const W = kind === "parade" ? 488 : kind === "vectorscope" ? 256 : 320, H = kind === "histogram" ? 200 : 256;
  const o = LIVE.out; o.width = W; o.height = H;
  const ox = o.getContext("2d"), id = ox.createImageData(W, H), d = id.data;
  for (let i = 0; i < W * H; i++) { d[4 * i] = BG[0]; d[4 * i + 1] = BG[1]; d[4 * i + 2] = BG[2]; d[4 * i + 3] = 255; }
  const paint = (hist, w, h, x0, col) => {            // log-scaled 2-D histogram -> colour, like scopes.py
    let mx = 0; for (let i = 0; i < hist.length; i++) if (hist[i] > mx) mx = hist[i];
    const lm = Math.log1p(mx) || 1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = hist[y * w + x]; if (!v) continue;
      const k = Math.log1p(v) / lm, c = typeof col === "function" ? col(x, y) : col, j = 4 * (y * W + x0 + x);
      d[j] = Math.min(255, BG[0] + k * c[0]); d[j + 1] = Math.min(255, BG[1] + k * c[1]); d[j + 2] = Math.min(255, BG[2] + k * c[2]);
    }
  };
  const wave = (w, h, val) => {
    const hist = new Float32Array(w * h);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const j = 4 * (y * sw + x), v = Math.max(0, Math.min(255, val(px[j], px[j + 1], px[j + 2]) | 0));
      hist[(((255 - v) * h / 256) | 0) * w + ((x * w / sw) | 0)] += 1;
    }
    return hist;
  };
  if (kind === "waveform") paint(wave(W, H, (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b), W, H, 0, [90, 230, 140]);
  else if (kind === "parade") {
    [[0, [255, 80, 80]], [1, [80, 230, 110]], [2, [90, 140, 255]]].forEach(([c, col], i) => paint(wave(160, H, (r, g, b) => [r, g, b][c]), 160, H, i * 164, col));
  } else if (kind === "vectorscope") {
    const S2 = 256, half = 128, sc = half / 128 * 0.95, hist = new Float32Array(S2 * S2);
    for (let j = 0; j < px.length; j += 4) {
      const r = px[j], g = px[j + 1], b = px[j + 2];
      const cb = -0.168736 * r - 0.331264 * g + 0.5 * b, cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
      const x = Math.max(0, Math.min(S2 - 1, (half + cb * sc) | 0)), y = Math.max(0, Math.min(S2 - 1, (half - cr * sc) | 0));
      hist[y * S2 + x]++;
    }
    paint(hist, S2, S2, 0, (x, y) => {
      const u = (x - half) / (half * 0.95) * 128, v = (half - y) / (half * 0.95) * 128, cl = (q) => Math.max(40, Math.min(255, q));
      return [cl(128 + 1.402 * v), cl(128 - 0.344 * u - 0.714 * v), cl(128 + 1.772 * u)];
    });
  } else {                                             // histogram
    [[0, [255, 70, 70]], [1, [70, 220, 100]], [2, [80, 130, 255]]].forEach(([c, col]) => {
      const cnt = new Float32Array(256); for (let j = c; j < px.length; j += 4) cnt[px[j]]++;
      let mx = 0; for (let i = 0; i < 256; i++) { cnt[i] = Math.sqrt(cnt[i]); if (cnt[i] > mx) mx = cnt[i]; }
      for (let i = 0; i < 256; i++) {
        const top = Math.round(H - cnt[i] / (mx || 1) * (H - 4)), x0 = Math.floor(i * W / 256);
        for (let y = top; y < H; y++) for (let x = x0; x < Math.min(W, x0 + 2); x++) {
          const q = 4 * (y * W + x); d[q] = Math.min(255, d[q] + col[0] * 0.45); d[q + 1] = Math.min(255, d[q + 1] + col[1] * 0.45); d[q + 2] = Math.min(255, d[q + 2] + col[2] * 0.45);
        }
      }
    });
  }
  ox.putImageData(id, 0, 0);
  ox.strokeStyle = "rgb(45,50,62)"; ox.lineWidth = 1;
  if (kind === "vectorscope") {
    for (const rr of [0.25, 0.5, 0.75, 1]) { ox.beginPath(); ox.arc(128, 128, 128 * 0.95 * rr, 0, 2 * Math.PI); ox.stroke(); }
    ox.strokeStyle = "rgb(170,120,90)"; ox.beginPath(); ox.moveTo(128, 128);
    const a = 123 * Math.PI / 180; ox.lineTo(128 + Math.cos(a) * 121.6, 128 - Math.sin(a) * 121.6); ox.stroke();
  } else if (kind === "histogram") {
    ox.strokeStyle = "rgb(90,70,40)"; for (const v of [16, 235]) { const x = Math.round(v * W / 256) + 0.5; ox.beginPath(); ox.moveTo(x, 0); ox.lineTo(x, H); ox.stroke(); }
  } else {
    for (const v of [0, 64, 128, 191, 255]) { const y = Math.round((255 - v) / 255 * (H - 1)) + 0.5; ox.beginPath(); ox.moveTo(0, y); ox.lineTo(W, y); ox.stroke(); }
    ox.strokeStyle = "rgb(90,70,40)"; for (const v of [16, 235]) { const y = Math.round((255 - v) / 255 * (H - 1)) + 0.5; ox.beginPath(); ox.moveTo(0, y); ox.lineTo(W, y); ox.stroke(); }
  }
  $("scopeImg").src = o.toDataURL("image/png");
  $("scopeWait").classList.add("hidden");
}
function initScopes() {
  for (const b of $("scopeSeg").children) b.onclick = () => { SCOPE.kind = b.dataset.v; for (const x of $("scopeSeg").children) x.classList.toggle("on", x === b); requestScopes(); };
  $("scopeSeg").children[0].classList.add("on");
}
// ---- LUT stack (grade.luts, applied top to bottom)
async function loadLuts(paths) {
  const add = [];
  for (const p of paths || []) {
    let i; try { i = await rpc("lut_info", { path: p }); } catch { continue; }
    if (i.error) { status(i.error, "err"); log(i.error, "err"); continue; }
    add.push({ path: p, mix: 1, on: true, info: `${i.kind.toUpperCase()} ${i.size}` });
    log(`LUT added: ${basename(p)} (${i.kind.toUpperCase()}, ${i.size} points)`, "ok");
  }
  if (!add.length) return;
  if (!S.info) return;
  for (const lut of add) { const l = lutLayerNew(lut, -1, -1, 0); l.pending = true; }      // listed, not applied: it is placed when you drop it on the Timeline
  S.layerSel = S.layers.length - 1;
  layersChanged();
  status(`${add.length === 1 ? basename(add[0].path) : add.length + " LUTs"} added to the list - drag ${add.length === 1 ? "its dot" : "a dot"} onto the Timeline to use ${add.length === 1 ? "it" : "one"} there (${rangeDefaultFrames()} frames = 1/30 of the video, stretch it afterwards).`, "ok");
}
function layerPending() {                            // "+ Layer": a colour adjustment layer waiting in the list (edit it now, drop it on the Timeline to use it)
  if (!S.info) return;
  const n = S.layers.length + 1;
  S.layers.push({ name: `Colour adjustment layer ${n}`, track: 0, start: -1, end: -1, on: true, opacity: 1, pending: true, grade: structuredClone(GRADE0), col: LAYER_COL[(n - 1) % LAYER_COL.length] });
  S.layerSel = S.layers.length - 1;
  layersChanged();
  status(`Colour adjustment layer ${n} added to the list - drag its dot onto the Timeline to use it there (${rangeDefaultFrames()} frames = 1/30 of the video, stretch it afterwards).`, "ok");
}
function lutLayerNew(lut, a, b, row) {               // a colour adjustment layer that carries one LUT
  const n = S.layers.length + 1, grade = structuredClone(GRADE0);
  grade.luts = [lut];
  const l = { name: `Colour adjustment layer ${n} · ${basename(lut.path)}`, track: row || 0, start: a, end: b, on: true, opacity: 1, grade, col: LAYER_COL[(n - 1) % LAYER_COL.length] };
  S.layers.push(l);
  return l;
}
function convertBaseLuts() {                          // LUTs kept in a grade's own list (base grade, a colour layer) become LUT layers: one list of layers
  if (!S.info || !S.gradeAll) return;
  let did = false;
  const g = S.gradeAll;
  if (g.lut) { g.luts = [{ path: g.lut, mix: g.lut_mix ?? 1, on: true }, ...(g.luts || [])]; g.lut = ""; }
  const luts = (g.luts || []).map((l) => ({ ...l })), [a, b] = epRange();
  if (luts.length && b > a) { for (const lut of luts) lutLayerNew(lut, a, b, 0); g.luts = []; did = true; }
  for (const l of [...S.layers]) {                    // a colour layer's own LUTs: layers of the same range just above it
    if (!l.grade || !(l.grade.luts || []).length || isLutLayer(l)) continue;
    l.grade.luts.map((x) => ({ ...x })).forEach((lut, k) => { const n = lutLayerNew(lut, l.start, l.end, (l.track || 1) + 0.5 + k * 0.01); n.on = l.on; });
    l.grade.luts = []; did = true;
  }
  if (did) setTimeout(layersChanged, 0);
}
// the colour of a bar: colour adjustment layers one colour, LUT layers another (every LUT alike), like a filter's ranges
const CAL_COL = "#a35fd0", LUT_COL = "#58c4a8";
function isLutLayer(l) { const g = (l && l.grade) || {}; return (g.luts || []).length > 0 && sameGrade({ ...g, luts: [] }, { ...GRADE0, luts: [] }); }
function layerColour(l) { return isLutLayer(l) ? LUT_COL : CAL_COL; }
let LUTL_SIG = "";
function lutLayersUi() {                              // in the LUTs window: the LUT layers of this episode (click one to edit its strength)
  const box = $("lutList");
  let host = $("lutLayers");
  if (!host) { host = el("div", "lut-layers"); host.id = "lutLayers"; box.parentNode.insertBefore(host, box); }
  const ls = epLayers().filter(({ l }) => isLutLayer(l)), sig = JSON.stringify([ls.map(({ l, i }) => [i, l.name, l.start, l.end, l.on]), S.layerSel]);
  if (sig === LUTL_SIG) return;
  LUTL_SIG = sig; host.textContent = "";
  if (!ls.length) return;
  host.appendChild(el("div", "note", "LUT layers (they are on the Timeline: click one to edit its strength here, drag it there to move or stretch it):"));
  for (const { l, i } of ls) {
    const b = el("button", "lut-layer" + (i === S.layerSel ? " sel" : ""));
    const sw = el("span", "sw"); sw.style.background = LUT_COL;
    b.append(sw, el("span", null, l.name.replace(/^Colour adjustment layer \d+ · /, "")), el("small", null, `${l.end - l.start} fr`));
    b.onclick = () => selectLayer(i);
    host.appendChild(b);
  }
}
function lutList() {
  const g = S.grade;
  if (g.lut) { g.luts = [{ path: g.lut, mix: g.lut_mix ?? 1, on: true }, ...(g.luts || [])]; g.lut = ""; }   // older settings
  return (g.luts || []).map((l) => ({ ...l }));
}
let LUT_SIG = "";
function lutUi() {
  lutLayersUi();
  const L = lutList(), sig = JSON.stringify(L);
  $("lutCount").textContent = L.length ? `${L.filter((l) => l.on !== false).length} of ${L.length} on` : "";
  if (sig === LUT_SIG && $("lutList").childElementCount === L.length) return;
  LUT_SIG = sig;
  const box = $("lutList"); box.innerHTML = "";
  const set = (arr) => changed("grade.luts", arr);
  L.forEach((l, i) => {
    const row = el("div", "mrow" + (l.on === false ? " off" : ""));
    const num = el("div", "num", String(i + 1)); num.draggable = true; num.title = "Drag onto the Timeline: this LUT then works only on that part";
    num.ondragstart = (e) => { e.dataTransfer.setData("application/x-eva-lut", String(i)); e.dataTransfer.effectAllowed = "copyMove"; };
    row.append(num);
    const mid = el("div");
    const nm = el("div", "mname", basename(l.path)); nm.title = l.path;
    mid.append(nm, el("div", "minfo", l.info || "LUT"));
    const line = el("div", "inline");
    const r = el("input"); r.type = "range"; r.min = 0; r.max = 1; r.step = 0.01; r.value = l.mix ?? 1;
    const t = el("input"); t.type = "text"; t.value = (+(l.mix ?? 1)).toFixed(2);
    const put = (v) => { const a = lutList(); a[i].mix = Math.max(0, Math.min(1, +(+v).toFixed(2))); set(a); };
    r.oninput = () => { t.value = (+r.value).toFixed(2); };
    r.onchange = () => put(r.value);
    t.onchange = () => { const v = parseFloat(t.value.replace(",", ".")); put(isNaN(v) ? l.mix ?? 1 : v); };
    r.ondblclick = () => put(1);
    line.append(el("span", null, "strength"), r, t);
    mid.append(line);
    const btns = el("div", "mbtns");
    const b = (txt, title, fn, dis) => { const x = el("button", null, txt); x.title = title; x.disabled = !!dis; x.onclick = fn; btns.appendChild(x); };
    b(l.on === false ? "Off" : "On", "Switch this LUT on / off", () => { const a = lutList(); a[i].on = a[i].on === false; set(a); });
    b("↑", "Apply earlier", () => { const a = lutList(); [a[i - 1], a[i]] = [a[i], a[i - 1]]; set(a); }, i === 0);
    b("↓", "Apply later", () => { const a = lutList(); [a[i + 1], a[i]] = [a[i], a[i + 1]]; set(a); }, i === L.length - 1);
    b("✕", "Remove", () => { const a = lutList(); a.splice(i, 1); set(a); });
    row.append(mid, btns);
    box.appendChild(row);
  });
}

// ---- adjustment layers on tracks (source frames of the whole title, so they survive episode edits)
const LAYER_COL = ["#a35fd0", "#6f86f0", "#d9963f", "#4fb07e", "#c95d9e", "#3fa6b8", "#b9a43a"];
function epRange() { const e = S.episodes[S.current]; return e ? [e[0], e[1]] : [0, 0]; }
function epLayers() {
  for (const l of S.layers) { const m = /^Adjustment Layer (\d+)$/.exec(l.name || ""); if (m) l.name = `Colour adjustment layer ${m[1]}`; }   // the earlier default name
  const [a, b] = epRange(); return S.layers.map((l, i) => ({ l, i })).filter(({ l }) => l.start < b && l.end > a); }
function layersChanged() { sync(); renderLayerTrack(); schedulePreview(); scheduleScript(); }
function playhead() { return Math.round(S.src ?? headSrcFrame()); }
function trackFree(t, s, e, skip) { return !S.layers.some((l) => l !== skip && (l.track || 1) === t && l.start < e && l.end > s); }
function gapAt(t, f, skip) {                            // free range on track t around frame f
  const [a, b] = epRange();
  let lo = a, hi = b;
  for (const l of S.layers) {
    if (l === skip || (l.track || 1) !== t) continue;
    if (l.start <= f && l.end > f) return null;
    if (l.end <= f) lo = Math.max(lo, l.end); else hi = Math.min(hi, l.start);
  }
  return [lo, hi];
}
function addLayer(track, at) {
  if (!S.info) return;
  const [a, b] = epRange();
  const f = Math.max(a, Math.min(b - 2, at ?? playhead())), len = Math.max(2, rangeDefaultFrames());
  let t = track, g = null;
  if (t) g = gapAt(t, f);
  else for (t = 1; t <= S.tracks; t++) { g = gapAt(t, f); if (g && g[1] - f >= 2) break; }
  if (!g || g[1] - f < 2) { if (track) return; t = ++S.tracks; g = [a, b]; }
  const n = S.layers.length + 1;
  S.layers.push({ name: `Colour adjustment layer ${n}`, track: t, start: f, end: Math.min(g[1], f + len), on: true, opacity: 1,
                  grade: structuredClone(GRADE0), col: LAYER_COL[(n - 1) % LAYER_COL.length] });
  S.layerSel = S.layers.length - 1; RPIN = S.layers[S.layerSel];
  layersChanged();
}
// Dropped on the Timeline from the Colour tab: "+ Layer" = a new colour adjustment layer, a layer row (its handle) = that layer moved there,
// a LUT row (its handle) = a layer that carries the LUT (the LUT leaves the list it was in). 1/30 of the video from the drop point on the layer
// under the mouse; the usual layer rules then settle what it overlaps.
function layerAtDrop(f, row, lut) {
  if (!S.info) return null;
  const [a, b] = epRange(), len = Math.max(2, Math.min(rangeDefaultFrames(), b - a)), start = Math.max(a, Math.min(f, b - len));
  const n = S.layers.length + 1, grade = structuredClone(GRADE0);
  if (lut) grade.luts = [lut];
  const l = { name: `Colour adjustment layer ${n}` + (lut ? ` · ${basename(lut.path)}` : ""), track: row || 0, start, end: start + len, on: true, opacity: 1,
              grade, col: LAYER_COL[(n - 1) % LAYER_COL.length] };       // (the bar's colour comes from layerColour)
  S.layers.push(l); S.layerSel = S.layers.length - 1; RPIN = l;
  layersChanged();
  return l;
}
function layerMoveTo(i, f, row) {
  const l = S.layers[i]; if (!l) return;
  const [a, b] = epRange(), wasPending = !!l.pending, len = wasPending ? Math.min(rangeDefaultFrames(), b - a) : l.end - l.start;
  l.start = Math.max(a, Math.min(f, b - len)); l.end = l.start + len;
  if (row) l.track = row; else if (wasPending) l.track = 0;
  delete l.pending;
  S.layerSel = i; RPIN = l;
  layersChanged();
  status(wasPending ? `${l.name} is on the Timeline now (${len} frames): stretch or move the bar.` : `${l.name} moved to ${rowLabel(l.track)}`, "ok");
}
function lutToTimeline(i, f, row) {
  const arr = lutList(), [lut] = arr.splice(i, 1);
  if (!lut) return;
  changed("grade.luts", arr);                          // out of the grade it was in (the base grade or the layer being edited) ...
  const l = layerAtDrop(f, row, lut);                  // ... into a layer of its own
  if (l) status(`${basename(lut.path)} now works only on this part (${rangeDefaultFrames()} frames = 1/30 of the video): stretch or move the bar on the Timeline.`, "ok");
}
function splitLayer() {
  const f = playhead();
  let i = S.layerSel;
  const inside = (l) => l && f > l.start && f < l.end;
  if (!inside(S.layers[i])) {                          // the top layer under the playhead
    i = -1;
    S.layers.forEach((l, k) => { if (inside(l) && (i < 0 || (l.track || 1) > (S.layers[i].track || 1))) i = k; });
  }
  if (i < 0) { status("No layer under the playhead to split."); return; }
  const l = S.layers[i], r = structuredClone(l);
  r.start = f; l.end = f; r.name = l.name.replace(/( \(\d+\))?$/, "") + " (2)";
  S.layers.splice(i + 1, 0, r);
  S.layerSel = i + 1;
  layersChanged();
}
async function selectLayer(i) {
  S.layerSel = i;
  sync(); renderLayerTrack();
  const l = S.layers[i];
  if (!l || l.pending || S.src == null || (S.src >= l.start && S.src < l.end)) return;
  try {                                              // the frame shown is outside the layer: go to it
    const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: Math.max(l.start, epRange()[0]) + 2 });
    togglePlay(false); S.frame = r.frame; schedulePreview();
  } catch { /* logged */ }
}
function moveTrack(i, d) {                          // the arrows in the Colour tab: one layer up / down (what it overlaps moves away)
  const l = S.layers[i], t = (l.track || 1) + d;
  if (t < 1) return;
  l.track = t; RPIN = l; layersChanged();
}
function trimTracks() { normalizeTracks(); }
// Adjustment layers follow the same rules as the Timeline's filter layers: any layer can go on any track (what it overlaps then moves
// to the first track where it fits), above the top / below the bottom track makes a new one, empty tracks close up.
const resolveTracks = (pin) => settleRows(pin);
const normalizeTracks = () => settleRows();
function removeTrack(t) {                          // the x on a layer's head: everything on it
  const name = rowLabel(t);
  const gone = S.layers.filter((l) => !l.pending && (l.track || 1) === t), ranges = activeRangeItems().filter((x) => (x.r.layer || 1) === t);
  if (gone.some((l) => !sameGrade(l.grade, GRADE0) && !isLutLayer(l)) && !confirm(`Delete ${name} and its ${gone.length} colour layer${gone.length > 1 ? "s" : ""} with their grades?`)) return;
  const sel = S.layers[S.layerSel];
  S.layers = S.layers.filter((l) => !gone.includes(l));
  S.layerSel = sel && !gone.includes(sel) ? S.layers.indexOf(sel) : -1;
  if (ranges.length) { rangesDelete(ranges); }          // says what is left without a range
  settleRows(); layersChanged();
  if (!ranges.length) status(gone.length ? `${name} deleted (${gone.length} colour layer${gone.length > 1 ? "s" : ""})` : `${name} closed`, "ok");
}
function layerMenu(e, i) {                         // right-click on an adjustment layer
  e.preventDefault(); e.stopPropagation();
  const l = S.layers[i]; if (!l) return;
  S.layerSel = i; sync(); renderLayerTrack();
  closeRangeMenu();
  const m = el("div", "ctx-menu"); m.id = "rngMenu";
  const add = (t, fn) => { const b = el("button", null, t); b.onclick = () => { closeRangeMenu(); fn(); }; m.appendChild(b); };
  add(`Delete ${l.name}`, () => removeLayer(i));
  add(`Delete ${rowLabel(l.track || 1)}`, () => removeTrack(l.track || 1));
  add("Copy (Ctrl+C)", () => layerCopy(i, false));
  add("Go to this layer", () => selectLayer(i));
  document.body.appendChild(m);
  m.style.left = Math.min(e.clientX, innerWidth - m.offsetWidth - 6) + "px"; m.style.top = Math.min(e.clientY, innerHeight - m.offsetHeight - 6) + "px";
  setTimeout(() => document.addEventListener("pointerdown", closeOnOutside, true), 0);
}
let LCLIP = null, CLIPKIND = "range";              // Ctrl+C / X / V on an adjustment layer (the last thing copied decides what V pastes)
function layerCopy(i, cut) {
  const l = S.layers[i]; if (!l) return;
  LCLIP = { layer: structuredClone(l), len: l.end - l.start, track: l.track || 1 }; CLIPKIND = "layer";
  if (cut) removeLayer(i, false); else status(`Copied ${l.name} (${LCLIP.len} frames) - Ctrl+V pastes it at the playhead`, "ok");
}
function layerPaste() {
  if (!LCLIP || !S.info) return;
  const [a, b] = epRange(), len = Math.min(LCLIP.len, b - a), start = Math.max(a, Math.min(playhead(), b - len));
  const l = structuredClone(LCLIP.layer), n = S.layers.length + 1;
  const suf = /( · .+)$/.exec(LCLIP.layer.name || "");     // a LUT layer keeps its LUT's name
  l.start = start; l.end = start + len; l.name = `Colour adjustment layer ${n}` + (suf ? suf[1] : ""); l.col = LAYER_COL[(n - 1) % LAYER_COL.length];
  let t = Math.max(1, LCLIP.track);
  while (t <= S.tracks && !trackFree(t, l.start, l.end, null)) t++;
  l.track = t;
  S.layers.push(l); S.layerSel = S.layers.length - 1;
  RPIN = l; layersChanged();
  status(`Pasted ${l.name}`, "ok");
}
function removeLayer(i, ask = true) {
  const l = S.layers[i];
  if (ask && !sameGrade(l.grade, GRADE0) && !isLutLayer(l) && !confirm(`Delete "${l.name}" and its grade?`)) return;
  S.layers.splice(i, 1);
  if (S.layerSel === i) S.layerSel = -1; else if (S.layerSel > i) S.layerSel--;
  trimTracks();
  layersChanged();
}
let LAYER_SIG = "";
function layerUi() {
  if (!(typeof TLX !== "undefined" && TLX.drag)) { convertBaseLuts(); settleRows(); }      // rows and LUT layers settled before they are listed
  const sel = S.layers[S.layerSel], list = epLayers();
  if (sel) {
    const out = !sel.pending && S.src != null && (S.src < sel.start || S.src >= sel.end);
    $("layerTarget").innerHTML = `Editing: <b>${sel.name.replace(/[<&]/g, "")}</b> · ` + (sel.pending ? '<span class="warn">not on the Timeline yet: drag its dot there</span>' : `${rowLabel(sel.track || 1)} · ${tc(sel.start)} – ${tc(sel.end)}`) +
      (!sel.pending && (sel.on === false || S.trackOff[sel.track || 1]) ? ' · <span class="warn">switched off</span>' : "") + (out ? ' · <span class="warn">the frame shown is outside this layer</span>' : "");
  } else {
    $("layerTarget").innerHTML = "Editing: <b>Base grade</b> (whole video)" + (list.length ? ` · ${list.length} adjustment layer${list.length === 1 ? "" : "s"} on top` : "");
  }
  $("btnGradeReset").textContent = sel ? "Reset layer" : "Reset all";
  $("layerCount").textContent = list.length ? `${list.length} on this episode` : "";
  $("layerTarget").classList.toggle("layer", !!sel);
  $("btnLayerAdd").disabled = !S.info;
  const addBtn = $("btnLayerAdd");
  if (!addBtn.draggable) {                         // drag it onto the Timeline for a new layer where you drop it
    addBtn.draggable = true; addBtn.title = "New colour adjustment layer at the playhead - or drag this button onto the Timeline";
    addBtn.ondragstart = (e) => { e.dataTransfer.setData("application/x-eva-clayer-new", "1"); e.dataTransfer.effectAllowed = "copy"; };
  }
  const sig = JSON.stringify([list.map(({ l, i }) => [i, l.name, l.track, l.start, l.end, l.on, l.opacity, l.pending, isLutLayer(l), ((l.grade && l.grade.luts) || []).map((x) => x.mix).join()]), S.layerSel, S.current, S.layers.length, S.trackOff]);
  if (sig === LAYER_SIG) return;
  LAYER_SIG = sig;
  const box = $("layerList"); box.innerHTML = "";
  const waiting = S.layers.map((l, i) => ({ l, i })).filter(({ l }) => l.pending);          // in the list, not on the Timeline yet
  const sorted = [...list].sort((p, q) => (q.l.track || 1) - (p.l.track || 1) || p.l.start - q.l.start);   // top layer first
  sorted.push(...waiting);
  for (const { l, i } of sorted) {
    const lut = isLutLayer(l), isSel = i === S.layerSel;
    const row = el("div", "layer-row lr2" + (isSel ? " sel" : "") + (l.pending ? " waiting" : "") + (!l.pending && (l.on === false || S.trackOff[l.track || 1]) ? " off" : ""));
    row.draggable = true;                              // the whole row drags onto the Timeline; text boxes keep their own mouse
    row.addEventListener("mousedown", (e) => { row.draggable = !e.target.closest("input, textarea, select"); });
    row.addEventListener("dragstart", (e) => { if (e.target !== row) return; e.dataTransfer.setData("application/x-eva-clayer", String(i)); e.dataTransfer.effectAllowed = "copyMove"; });
    const top = el("div", "lr-top");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = l.on !== false; cb.title = "On / off";
    cb.onclick = (e) => e.stopPropagation();
    cb.onchange = () => { l.on = cb.checked; layersChanged(); };
    const sw = el("span", "sw"); sw.style.background = layerColour(l);
    sw.draggable = true; sw.title = l.pending ? "Drag onto the Timeline to use this layer there" : "Drag onto the Timeline to move this layer there";
    sw.ondragstart = (e) => { e.dataTransfer.setData("application/x-eva-clayer", String(i)); e.dataTransfer.effectAllowed = "copyMove"; };
    const nm = el("input", "lname"); nm.type = "text"; nm.value = l.name; nm.title = `${l.name} - click to rename`;
    nm.onchange = () => { l.name = nm.value.trim() || l.name; layersChanged(); };
    nm.onclick = (e) => { e.stopPropagation(); if (S.layerSel !== i) selectLayer(i); };
    const num = (v, tip) => { const x = el("input", "lr-num mono"); x.type = "number"; x.min = "0"; x.step = "1"; x.value = String(v); x.title = tip; x.onclick = (e) => e.stopPropagation(); return x; };
    const A = num(Math.max(0, l.start), "First frame of the range (source frame number)"), B = num(Math.max(0, l.end - 1), "Last frame of the range (source frame number)");
    const setRange = () => {
      const [ea, eb] = epRange(), p0 = parseInt(A.value, 10), p1 = parseInt(B.value, 10);
      if (Number.isNaN(p0) || Number.isNaN(p1)) { A.value = l.start; B.value = l.end - 1; return; }
      const s0 = Math.max(ea, Math.min(eb - 1, Math.min(p0, p1))), e0 = Math.max(s0 + 1, Math.min(eb, Math.max(p0, p1) + 1));
      l.start = s0; l.end = e0; RPIN = l; layersChanged();
    };
    A.onchange = B.onchange = setRange;
    const rng = el("span", "lr-range");
    if (l.pending) rng.append(el("span", "muted", "not on the Timeline - drag the dot there"));
    else rng.append(el("span", "muted", "Range"), A, el("span", "muted", "to"), B);
    const x = el("button", "ghost", "✕"); x.title = "Delete this layer"; x.onclick = (e) => { e.stopPropagation(); removeLayer(i); };
    top.append(cb, sw, nm, rng, x);
    row.append(top);
    if (isSel) {                                       // the one being edited: strength (LUT) / opacity, and the range from the frame shown
      const l2 = el("div", "l2");
      const mix = lut ? () => l.grade.luts[0].mix ?? 1 : () => l.opacity ?? 1;
      const r = el("input"); r.type = "range"; r.min = 0; r.max = 1; r.step = 0.01; r.value = mix();
      const t = el("input"); t.type = "text"; t.value = Math.round(100 * mix()) + "%";
      const put = (v) => { const q = Math.max(0, Math.min(1, +(+v).toFixed(2))); if (lut) l.grade.luts[0].mix = q; else l.opacity = q; layersChanged(); };
      r.oninput = () => { t.value = Math.round(100 * r.value) + "%"; };
      r.onchange = () => put(r.value);
      t.onchange = () => { const v = parseFloat(t.value); put(isNaN(v) ? mix() : v / 100); };
      r.ondblclick = () => put(1);
      for (const q of [r, t]) q.onclick = (e) => e.stopPropagation();
      const at = (txt, title, fn, dis) => { const b = el("button", null, txt); b.title = title; b.disabled = !!dis; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
      const here = S.src;
      if (l.pending) l2.append(el("span", null, lut ? "strength" : "opacity"), r, t);
      else l2.append(el("span", null, `${rowLabel(l.track || 1)} · ${l.end - l.start} fr · ${lut ? "strength" : "opacity"}`), r, t,
        at("Start here", "Layer starts at the frame shown", () => { if (here != null && here < l.end) { l.start = here; RPIN = l; layersChanged(); } }, here == null || here >= l.end),
        at("End here", "Layer ends after the frame shown", () => { if (here != null && here >= l.start) { l.end = here + 1; RPIN = l; layersChanged(); } }, here == null || here < l.start));
      row.append(l2);
    }
    row.onclick = () => selectLayer(i);
    box.appendChild(row);
  }
  const base = el("div", "layer-row base" + (S.layerSel < 0 ? " sel" : ""));
  const l1 = el("div", "l1"), sw = el("span", "sw"); sw.style.background = "#3b5378";
  const nm = el("input", "lname"); nm.value = "Base grade · whole video (V1)"; nm.readOnly = true; nm.tabIndex = -1;
  l1.append(sw, nm); base.append(l1);
  base.onclick = () => selectLayer(-1);
  box.appendChild(base);
  const other = S.layers.length - list.length - waiting.length;
  if (other) box.appendChild(el("div", "scene-empty", `${other} layer${other === 1 ? "" : "s"} in other episodes`));
}

// ---- the multi-track timeline under the picture (Colour step)

// The Colour tab's windows: Colour adjustment layers and LUTs first, every window movable by its handle (the order is remembered)
function initColourOrder() {
  const panel = document.querySelector('[data-panel="colour"]');
  if (!panel || panel._ordered) return;
  panel._ordered = true;
  const keyOf = (g) => g.dataset.fold || (g.id === "autoGroup" ? "auto" : g.id || "");
  const groups = () => [...panel.querySelectorAll(":scope > .group")];
  const DEFAULT = ["scopes", "auto", "layers", "luts", "wheels", "light", "colour", "more", "curves"];
  let saved = []; try { saved = JSON.parse(localStorage.getItem("colourOrder") || "[]"); } catch { /* none */ }
  const order = [...saved.filter((k) => DEFAULT.includes(k)), ...DEFAULT.filter((k) => !saved.includes(k))];
  const all = groups(), end = el("div"); end.hidden = true;
  all[all.length - 1].after(end);
  for (const k of order) { const g = all.find((x) => keyOf(x) === k); if (g) panel.insertBefore(g, end); }
  const line = el("div", "cg-dropline hidden"); panel.appendChild(line);
  const mime = "application/x-eva-cgroup";
  let drag = null;
  for (const g of all) {
    const head = g.querySelector(":scope > .group-head"); if (!head) continue;
    const grip = el("span", "cg-grip", "⠿"); grip.title = "Drag to move this window"; grip.draggable = true;
    grip.onclick = (e) => e.stopPropagation();
    grip.ondragstart = (e) => { drag = g; e.dataTransfer.setData(mime, keyOf(g)); e.dataTransfer.effectAllowed = "move"; g.classList.add("dragging"); };
    grip.ondragend = () => { drag = null; g.classList.remove("dragging"); line.classList.add("hidden"); };
    head.prepend(grip);
  }
  const target = (y) => groups().filter((x) => x !== drag).find((x) => { const r = x.getBoundingClientRect(); return y < r.top + r.height / 2; }) || null;
  panel.addEventListener("dragover", (e) => {
    if (!drag || !e.dataTransfer.types.includes(mime)) return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    const t = target(e.clientY), pr = panel.getBoundingClientRect(), anchor = t || groups().filter((x) => x !== drag).pop();
    const r = anchor.getBoundingClientRect();
    line.style.top = (panel.scrollTop + (t ? r.top : r.bottom) - pr.top - 2) + "px"; line.classList.remove("hidden");
  });
  panel.addEventListener("drop", (e) => {
    if (!drag || !e.dataTransfer.types.includes(mime)) return;
    e.preventDefault();
    const t = target(e.clientY);
    panel.insertBefore(drag, t || end);
    line.classList.add("hidden");
    try { localStorage.setItem("colourOrder", JSON.stringify(groups().map(keyOf))); } catch { /* none */ }
  });
}
function initColourUi() {
  initColourOrder();
  buildSliders(); buildWheels(); initCurves(); initScopes(); buildFolds(); initRef();
  $("btnLutLoad").onclick = async () => loadLuts(await window.api.openLut());
  $("btnLayerAdd").onclick = () => layerPending();
  initLayerTrack();
}
function syncColourUi() { lutUi(); layerUi(); findingsUi(); refUi(); syncSliders(); for (const w of $$(".wheel[data-wheel]")) if (w._bg) drawWheel(w); drawCurves(); }

// ---------------------------------------------------------------- workspace (index.html / workspace.css)
const DOCK = { tab: "timeline" };
function showLeft(which) {
  document.body.classList.remove("left-off", "viewer-full");
  for (const b of $$("[data-left]")) b.classList.toggle("on", b.dataset.left === which);
  $("left-media").classList.toggle("hidden", which !== "media");
  $("left-presets").classList.toggle("hidden", which !== "presets");
  if (which === "presets" && typeof renderPresets === "function") renderPresets();
  setTimeout(layoutImage, 0);
}
function showDock(which) {
  if (which === "script") {                        // no Script tab in the dock: the script is a window
    which = "timeline";
    setTimeout(() => scriptWinShow(true), 0);
  }
  document.body.classList.remove("dock-off", "viewer-full");
  DOCK.tab = which;
  for (const b of $$("[data-dock]")) b.classList.toggle("on", b.dataset.dock === which);
  for (const p of $$("[data-dpage]")) p.classList.toggle("hidden", p.dataset.dpage !== which);
  if (which === "script") refreshScript();
  if (which === "timeline") { renderTimeline(); renderLayerTrack(); }
  if (which === "queue" && typeof renderQueue === "function") renderQueue();
  setTimeout(layoutImage, 0);
}
// panel sizes (remembered) and the splitters between the panels
const WS_DEF = { left: 300, right: 400, dock: 320, v: 3 };
function wsLoad() {
  try {
    const saved = JSON.parse(localStorage.getItem("wsLayout") || "{}"), l = Object.assign({}, WS_DEF, saved);
    if ((saved.v || 1) < 2) { l.dock = Math.max(l.dock, 320); l.v = 2; }     // the editor Timeline needs more room than the old bar
    if ((saved.v || 1) < 3) { l.right = Math.min(l.right, 460); l.left = Math.min(l.left, 360); l.v = 3; }   // Script mode's wide Inspector was saved
    return l;
  } catch { return { ...WS_DEF }; }
}
function wsApply(l) {
  const a = $("app");
  a.style.setProperty("--left", l.left + "px"); a.style.setProperty("--right", l.right + "px"); a.style.setProperty("--dock", l.dock + "px");
  setTimeout(layoutImage, 0);
}
function wsSave(l) { try { localStorage.setItem("wsLayout", JSON.stringify(l)); } catch { /* no storage */ } }
function initSplitters() {
  const l = wsLoad(); wsApply(l);
  for (const sp of $$(".splitv, .splith")) {
    sp.addEventListener("pointerdown", (e) => {
      e.preventDefault(); sp.classList.add("drag"); sp.setPointerCapture(e.pointerId);
      const app = $("app").getBoundingClientRect(), work = $("work").getBoundingClientRect();
      const move = (ev) => {
        if (sp.dataset.split === "left") l.left = Math.round(Math.max(220, Math.min(560, ev.clientX - work.left)));
        else if (sp.dataset.split === "right") l.right = Math.round(Math.max(320, Math.min(720, work.right - ev.clientX)));
        else l.dock = Math.round(Math.max(110, Math.min(app.height * 0.6, app.bottom - ev.clientY - 30)));
        wsApply(l);
      };
      const up = () => { sp.classList.remove("drag"); sp.removeEventListener("pointermove", move); wsSave(l); layoutImage(); };
      sp.addEventListener("pointermove", move);
      sp.addEventListener("pointerup", up, { once: true });
    });
  }
}
const WS_CMD = {
  "toggle-left": () => { document.body.classList.toggle("left-off"); setTimeout(layoutImage, 0); },
  "toggle-right": () => { document.body.classList.toggle("right-off"); setTimeout(layoutImage, 0); },
  "script-win": () => scriptWinShow(!scriptWinOpen()),
  reload: () => reloadApp(),
  autosave: () => autosaveToggle(),
  "autosave-clear": () => autosaveForget(),
  "toggle-dock": () => { document.body.classList.toggle("dock-off"); setTimeout(layoutImage, 0); },
  fullscreen: () => { document.body.classList.toggle("viewer-full"); setTimeout(layoutImage, 0); },
  "layout-reset": () => { wsSave(WS_DEF); wsApply({ ...WS_DEF }); document.body.classList.remove("left-off", "right-off", "dock-off", "viewer-full"); },
  "preset-save": () => (typeof presetSave === "function" ? presetSave() : null),
  "queue-add": () => $("btnExport").click(),
  snapshot: () => (typeof snapshotTake === "function" ? snapshotTake() : null),
  bookmark: () => (typeof bookmarkAdd === "function" ? bookmarkAdd() : null),
};
function initMenus() {
  const close = () => { for (const m of $$(".menu.open")) m.classList.remove("open"); };
  for (const m of $$(".menu")) {
    m.querySelector(".mbtn").onclick = (e) => { e.stopPropagation(); const was = m.classList.contains("open"); close(); m.classList.toggle("open", !was); };
    m.querySelector(".mbtn").onmouseenter = () => { if ($$(".menu.open").length && !m.classList.contains("open")) { close(); m.classList.add("open"); } };
  }
  document.addEventListener("click", (e) => {
    const it = e.target.closest(".mlist button");
    if (it) {
      close();
      if (it.dataset.click) $(it.dataset.click).click();
      else if (it.dataset.cmd && WS_CMD[it.dataset.cmd]) WS_CMD[it.dataset.cmd]();
      return;
    }
    if (!e.target.closest(".menu")) close();
  });
  for (const b of $$("button[data-cmd]:not(.mlist button)")) b.addEventListener("click", () => WS_CMD[b.dataset.cmd] && WS_CMD[b.dataset.cmd]());
}
// Inspector: every Repair step one line (its name + what it is set to); clicking it opens only that step
function inspHeads() {
  for (const g of $$('.insp [data-panel="adjust"] > .group')) {
    const head = g.querySelector(":scope > .group-head"), h3 = head && head.querySelector(":scope > h3");
    if (!head || !h3 || head.querySelector(".ghead-txt")) continue;
    const box = el("div", "ghead-txt"); h3.replaceWith(box); box.append(h3, el("div", "gsum"));
    if (g.dataset.gid && STATIC_IDS.includes(g.dataset.gid)) {      // delete / restore this built-in step
      const x = el("button", "step-x", "✕"); const id = g.dataset.gid;
      x.onclick = (e) => {
        e.stopPropagation();
        const cur = S.removed_steps || [];
        const nx = cur.includes(id) ? cur.filter((s) => s !== id) : [...cur, id];
        changed("removed_steps", nx);
        log(`${STEP_NAME[id].replace(/^the /, "")}: ${nx.includes(id) ? "deleted - it runs nothing now" : "back"}`, "ok");
      };
      head.appendChild(x);
    }
    head.appendChild(el("span", "gcaret", "›"));
  }
}
function stepsRestoreAll() { changed("removed_steps", []); }
function fieldValue(f) {
  const sel = f.querySelector("select"), seg = f.querySelector(".seg"), step = f.querySelector(".stepper input, .slider input"), cb = f.querySelector('input[type="checkbox"]');
  if (seg) { const on = seg.querySelector("button.on"); return on ? on.textContent.trim() : ""; }
  if (sel) return sel.selectedOptions[0] ? sel.selectedOptions[0].textContent.replace(/\s*\(recommended\)/i, "").trim() : "";
  if (step) { const lab = (f.querySelector("label")?.childNodes[0]?.textContent || "").trim(); return `${lab} ${step.value}${f.querySelector(".stepper")?.dataset.unit === "px" ? " px" : ""}`; }
  if (cb) { const lab = (f.querySelector("label")?.childNodes[0]?.textContent || "").trim(); return cb.checked ? lab : ""; }
  return "";
}
function stepSummaries() {
  for (const g of $$('.insp [data-panel="adjust"] > .group')) {
    const out = g.querySelector(".gsum");
    if (!out) continue;
    const vals = [];
    const gone = !!g.dataset.gid && (S.removed_steps || []).includes(g.dataset.gid);
    g.classList.toggle("removed", gone);
    const ra = $("stepsRestoreAll"), da = $("stepsDeleteAll"), nrm = (S.removed_steps || []).length;
    if (ra) ra.classList.toggle("hidden", !nrm);
    if (da) da.classList.toggle("hidden", nrm >= STATIC_IDS.length);
    const xb = g.querySelector(".step-x");
    if (xb) { xb.textContent = gone ? "Restore" : "✕"; xb.title = gone ? "Put this step back" : "Delete this step: it runs nothing until you restore it (filters you placed after it still run)"; xb.classList.toggle("restore", gone); }
    if (gone) { g.classList.remove("open"); out.textContent = "Deleted - does nothing"; continue; }
    for (const f of g.querySelectorAll(":scope > .field, :scope > div > .field")) {
      if (f.closest(".hidden") || f.classList.contains("hidden") || f.closest(".rng-box")) continue;
      const v = fieldValue(f);
      if (v && !vals.includes(v)) vals.push(v);
      if (vals.length >= 3) break;
    }
    if (g.classList.contains("vsf-group")) { const n = g.querySelectorAll(".vk-card").length; vals.unshift(n ? `${n} filter${n > 1 ? "s" : ""}` : "empty"); }
    const rkey = RANGE_STEPS.includes(g.dataset.gid) ? g.dataset.gid : g.classList.contains("vsf-group") ? "vs:" + g.dataset.gid : null;
    if (rkey) {
      const n = rangeList(rkey).length, only = rkey.startsWith("vs:") ? (S.vs_groups.find((x) => x.id === rkey.slice(3)) || {}).only : (S.range_only || {})[rkey];
      if (only) vals.unshift(n ? `${n} range${n > 1 ? "s" : ""} only` : "ranges only (none set: off)");
    }
    out.textContent = vals.join(" · ");
  }
}
function initInspector() {
  inspHeads();
  const panel = document.querySelector('.insp [data-panel="adjust"]');
  let openId = null;
  try { openId = localStorage.getItem("inspOpen"); } catch { /* none */ }
  const idOf = (g) => g.dataset.gid || (g.classList.contains("vsf-group") ? "vsf:" + [...panel.querySelectorAll(".vsf-group")].indexOf(g) : "");
  const reopen = () => { for (const g of panel.querySelectorAll(":scope > .group")) g.classList.toggle("open", !!openId && idOf(g) === openId); };
  panel.addEventListener("click", (e) => {
    const head = e.target.closest(".group-head");
    if (!head || head.parentElement.parentElement !== panel) return;
    if (e.target.closest("button, input, select, label, .vsf-grip, .rep-grip, a")) return;
    const g = head.parentElement, id = idOf(g);
    openId = g.classList.contains("open") ? null : id;
    try { localStorage.setItem("inspOpen", openId || ""); } catch { /* no storage */ }
    reopen(); stepSummaries();
  });
  // filter groups are drawn again on every change: give them the same one-line head
  const orig = window.renderVsGroups;
  if (typeof orig === "function") window.renderVsGroups = renderVsGroups = function (...a) { const r = orig.apply(this, a); inspHeads(); renderRangeBoxes(); reopen(); stepSummaries(); applyFind(); return r; };
  renderRangeBoxes(); reopen(); stepSummaries();
  $("findSetting").addEventListener("input", applyFind);
  $("findSetting").addEventListener("keydown", (e) => { if (e.key === "Escape") { e.target.value = ""; applyFind(); } });
}
function applyFind() {
  const q = ($("findSetting")?.value || "").trim().toLowerCase();
  const panel = document.querySelector(`.insp [data-panel="${S.step}"]`);
  for (const g of $$(".insp .group")) g.classList.remove("found", "search-off");
  for (const f of $$(".insp .field.hit")) f.classList.remove("hit");
  if (!q || !panel) return;
  let any = 0;
  for (const g of panel.querySelectorAll(".group")) {
    if (g.parentElement.closest(".group")) continue;              // a group inside a group goes with its parent
    const hit = g.textContent.toLowerCase().includes(q);
    g.classList.toggle("found", hit); g.classList.toggle("search-off", !hit);
    if (hit) { any++; for (const f of g.querySelectorAll(".field")) if (f.textContent.toLowerCase().includes(q)) f.classList.add("hit"); }
  }
  if (!any) status(`No setting on this page matches "${q}".`);
}
function initWorkspace() {
  for (const b of $$("#pageTabs button")) b.onclick = () => goto(b.dataset.page);
  for (const b of $$("[data-left]")) b.onclick = () => showLeft(b.dataset.left);
  for (const b of $$("[data-dock]")) b.onclick = () => showDock(b.dataset.dock);
  $("btnQueue").onclick = () => showDock("queue");
  $("btnSnap").onclick = snapshotTake; $("btnBookmark").onclick = bookmarkAdd; $("btnFull").onclick = () => WS_CMD.fullscreen();
  $("abSel").onchange = () => { S.abSnap = $("abSel").value; schedulePreview(30); };
  initMenus(); initSplitters(); initInspector(); initPresets();
  window.addEventListener("resize", () => setTimeout(layoutImage, 0));
  window.addEventListener("keydown", (e) => {
    const typing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "range";
    if ((e.ctrlKey || e.metaKey) && (e.key === "o" || e.key === "O")) { e.preventDefault(); $("btnOpenFile").click(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === "f") { e.preventDefault(); document.body.classList.remove("right-off"); $("findSetting").focus(); return; }
    if (typing || e.ctrlKey || e.altKey || e.metaKey) return;
    if (e.key === "1" || e.key === "2" || e.key === "3") { const p = ["adjust", "colour", "export"][+e.key - 1]; if (p === "adjust" || S.info) { e.preventDefault(); goto(p); } }
    else if (e.key === "f" || e.key === "F") { e.preventDefault(); WS_CMD.fullscreen(); }
    else if ((e.key === "m" || e.key === "M") && S.info) { e.preventDefault(); bookmarkAdd(); }
    else if (e.key === "Escape" && document.body.classList.contains("viewer-full")) WS_CMD.fullscreen();
  });
}

// ---------------------------------------------------------------- frame ranges (a step or filter group from frame A to B)
// every range has a number inside its step / group ("Compression R1") and a colour of its own (the palette's first unused one)
const RANGE_PALETTE = ["#2fa37a", "#6f9a2f", "#d2504a", "#a35cba", "#5b8fe0", "#d1768a", "#8f96a6", "#b0743a"];   // for filters of your own (by name: the same filter = the same colour)
const allRanges = () => [...(S.ranges || []), ...(S.vs_groups || []).flatMap((g) => g.ranges || [])];
function rangeBase(key) {                          // the name a range of this step / group carries
  if (!key.startsWith("vs:")) return RANGE_NAME[key] || key;
  const g = S.vs_groups.find((x) => x.id === key.slice(3));
  return g && g.filters[0] ? g.filters[0].label + (g.filters.length > 1 ? ` +${g.filters.length - 1}` : "") : "Filters";
}
function rangeTitle(key, r) {
  const n = r.n || rangeList(key).indexOf(r) + 1;
  return `${rangeBase(key)} R${n}`;
}
function rangeColor(key) {                         // one colour per step; a filter of yours by its name
  if (!key.startsWith("vs:")) return RANGE_COL[key] || "#3b82d6";
  const g = S.vs_groups.find((x) => x.id === key.slice(3));
  let h = 0; for (const c of String((g && g.filters[0] && g.filters[0].label) || key)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return RANGE_PALETTE[h % RANGE_PALETTE.length];
}
// Timeline layers (Layer 1, Layer 2 ...): any filter, step or model can sit on any layer as long as two ranges on one layer
// do not overlap in time. A range has r.layer; ensureLayers() fills in the missing ones (first layer where it fits), moves a
// range that now overlaps another on its layer, and closes the gaps.
function activeRangeItems() {                      // [{key, r}] of the steps / groups that run on ranges
  const out = [];
  for (const st of RANGE_STEPS) if ((S.range_only || {})[st]) for (const r of rangeList(st)) out.push({ key: st, r });
  for (const g of S.vs_groups || []) if (g.only) for (const r of g.ranges || []) out.push({ key: "vs:" + g.id, r });
  return out;
}
const rOverlap = (a, b) => a.start < b.end && b.start < a.end;
let RPIN = null;                                   // the bar just moved / added / pasted (a layer or a range): it keeps its row, what it overlaps moves away
// ONE set of layers holds both the colour adjustment layers (l.track) and the filter ranges (r.layer): two bars on a layer may not overlap
// in time. Layer 1 is next to V1, higher = on top (for colour layers also the order they are applied in).
function rowItems() {
  const out = S.layers.filter((l) => !l.pending).map((l) => ({ o: l, k: "c" }));
  for (const it of activeRangeItems()) out.push({ o: it.r, k: "r", key: it.key });
  return out;
}
const rowOf = (it) => (it.k === "c" ? it.o.track : it.o.layer);
const setRow = (it, v) => { if (it.k === "c") it.o.track = v; else it.o.layer = v; };
function settleRows(pin) {
  pin = pin || RPIN;
  const items = rowItems();
  const rank = (x) => (x.o === pin ? -1e9 : rowOf(x) || (x.k === "c" ? 1 : 1e9));
  const order = [...items].sort((a, b) => (rank(a) - rank(b)) || (a.o.start - b.o.start));
  const placed = {};
  for (const it of order) {
    const free = (t) => !(placed[t] || []).some((x) => x.o.start < it.o.end && x.o.end > it.o.start);
    let t = rowOf(it) || 0;
    if (!t || !free(t)) { t = 1; while (!free(t)) t++; }
    (placed[t] = placed[t] || []).push(it); setRow(it, t);
  }
  const used = [...new Set(items.map(rowOf))].sort((a, b) => a - b);        // close the gaps; the eye follows its layer
  const off = {};
  used.forEach((t, k) => { if (S.trackOff[t]) off[k + 1] = true; });
  for (const it of items) setRow(it, used.indexOf(rowOf(it)) + 1);
  S.tracks = Math.max(1, used.length); S.trackOff = off;
  RPIN = null;
  return items.filter((x) => x.k === "r");
}
const ensureLayers = () => settleRows();
function rowLabel(t) {                             // "Layer 3" while empty; "Filter layer", "Colour layer" or "Filter+Colour layer" by what is on it
  const c = epLayers().some(({ l }) => (l.track || 1) === t), f = activeRangeItems().some((x) => (x.r.layer || 1) === t);
  return `${c && f ? "Filter+Colour " : f ? "Filter " : c ? "Colour " : ""}${c || f ? "layer" : "Layer"} ${t}`;
}
function rangeLabel(r) { return `frames ${r.start} → ${r.end - 1} (${r.end - r.start})`; }   // the last frame included, as in the From / To boxes
function rangeDefaultFrames() { const [a, b] = epRange(); return Math.max(10, Math.round((b - a) / 30)); }   // a range dropped on the Timeline / added with + Range: 1/30 of the episode (at least 10 frames), stretched on the Timeline
const RANGE_STEPS = ["composite", "compression", "halos", "colour", "stab", "sharpen"];
const RANGE_KEYS = { composite: ["crawl", "rainbow", "composite"], compression: ["deblock"], halos: ["ghost", "ghost_strength", "halo_method"], colour: ["sx", "sy", "bleed", "warp", "neutral"],
                     stab: ["stab", "stab_level", "tfix_strength"], sharpen: ["sharp"] };
const RANGE_NAME = { composite: "Dots & rainbows", compression: "Compression", halos: "Halos", colour: "Colour align.", stab: "Stabilizer", sharpen: "Sharpen" };
const RANGE_COL = { composite: "#2aa5b8", compression: "#3b82d6", halos: "#e0833a", colour: "#8f63c9", stab: "#c9a227", sharpen: "#c0506f", vs: "#7a6a2c" };
const playSrc = () => (S.src != null ? S.src : Math.round(headSrcFrame()));
function rangeList(key) {                          // key: a step name or "vs:<group id>"
  if (key.startsWith("vs:")) { const g = S.vs_groups.find((x) => x.id === key.slice(3)); return g ? (g.ranges = g.ranges || []) : []; }
  return (S.ranges || []).filter((r) => r.step === key);
}
function rangesChanged(key) {
  if (key && key.startsWith("vs:")) vsgChanged();
  else changed("ranges", S.ranges);
  renderRangeBoxes(); renderLayerTrack();
}
function rangeAdd(key, from, to, layer) {                  // from / to: source frame numbers typed in the boxes (else 1/30 of the video at the playhead)
  if (!S.info) return;
  const [e0, e1] = S.episodes[S.current];
  let a = from ?? playSrc();
  const len = rangeDefaultFrames();
  if (from != null && to == null) a = Math.max(e0, Math.min(a, e1 - len));     // dropped near the end: the range fits inside
  const r = { id: "r" + Date.now().toString(36), start: Math.max(e0, a), end: Math.min(e1, to != null ? to + 1 : a + len), on: true, settings: {} };
  r.n = Math.max(0, ...rangeList(key).map((x, i) => x.n || i + 1)) + 1;
  if (layer) r.layer = layer;                        // dropped on this Timeline layer (or the first one it fits on)
  if (r.end - r.start < 2) { status(from != null ? `Frames ${from} to ${to}: not inside this episode (${e0} to ${e1 - 1}), or To is not after From.` : "The playhead is at the end of the episode - move it back to add a range there.", "err"); return; }
  if (key.startsWith("vs:")) { const g = S.vs_groups.find((x) => x.id === key.slice(3)); if (!g) return; g.ranges = [...(g.ranges || []), r]; g.only = true; }
  else { r.step = key; S.ranges = [...(S.ranges || []), r]; }
  S.rangeSel = key.startsWith("vs:") ? null : r.id;
  rangesChanged(key);
  status(`${rangeTitle(key, r)}: ${rangeLabel(r)} added${key.startsWith("vs:") ? "" : " - its own settings: change them now with this step's controls"}. Drag it on the timeline.`, "ok");
}
function rangeSetEdge(key, edge) {
  const rs = rangeList(key);
  let r = rs.find((x) => x.id === S.rangeSel) || rs[rs.length - 1];
  if (!r) { rangeAdd(key); return; }
  const f = playSrc();
  if (edge === "a") r.start = Math.min(f, r.end - 1); else r.end = Math.max(f + 1, r.start + 1);
  rangesChanged(key);
}
function rangeRemove(key, r) {
  if (key.startsWith("vs:")) { const g = S.vs_groups.find((x) => x.id === key.slice(3)); if (g) g.ranges = (g.ranges || []).filter((x) => x !== r); }
  else S.ranges = S.ranges.filter((x) => x !== r);
  if (S.rangeSel === r.id) S.rangeSel = null;
  rangesChanged(key); sync();
}
function rangeEdit(r) {                            // its own settings: the step's controls switch to it (again = the episode's)
  S.rangeSel = S.rangeSel === r.id ? null : r.id;
  renderRangeBoxes(); renderLayerTrack(); sync();
  if (S.rangeSel) { gotoSrc(r.start + Math.min(12, Math.floor((r.end - r.start) / 2))); status(`Editing ${rangeTitle(r.step, r)} (${rangeLabel(r)}): this step's controls now set its own settings.`); }
  else status("Back to the episode's own settings.");
}
function renderRangeBoxes() {
  for (const g of $$('.insp [data-panel="adjust"] > .group')) {
    const gid = g.dataset.gid || "";
    const key = RANGE_STEPS.includes(gid) ? gid : g.classList.contains("vsf-group") ? "vs:" + gid : null;
    let box = g.querySelector(":scope > .rng-box");
    if (!key) { if (box) box.remove(); continue; }
    if (!box) { box = el("div", "rng-box"); g.appendChild(box); }
    box.innerHTML = "";
    const isVs = key.startsWith("vs:"), grp = isVs ? S.vs_groups.find((x) => x.id === key.slice(3)) : null;
    const only = isVs ? !!(grp && grp.only) : !!(S.range_only || {})[key];
    const seg = el("div", "seg rng-seg");
    for (const [v, t] of [[false, "Whole episode"], [true, "Ranges"]]) {
      const b = el("button", v === only ? "on" : "", t);
      b.onclick = () => { if (isVs) { if (grp) grp.only = v; } else S.range_only = { ...(S.range_only || {}), [key]: v }; rangesChanged(key); };
      seg.appendChild(b);
    }
    const f = el("div", "field"); f.append(el("label", null, "Applies to"), seg); box.appendChild(f);
    if (!only) { g.classList.toggle("editing-range", false); continue; }   // the whole episode: no ranges (they are kept, and come back with Specific ranges)
    if (!rangeList(key).length) box.appendChild(el("div", "note", "No range yet: the step is off everywhere until you add one. Type the first and last frame below, or drag the step onto the Timeline."));
    for (const r of rangeList(key)) {
      const row = el("div", "rng" + (S.rangeSel === r.id ? " sel" : ""));
      const dot = el("span", "rdot"); dot.style.background = rangeColor(key);
      const own = !isVs && r.settings && Object.keys(r.settings).length;
      const main = el("div", "rng-main"); main.append(el("b", "rng-name", rangeTitle(key, r)), el("span", "mono", rangeLabel(r) + (own ? " · own settings" : "")));
      row.append(dot, main);
      if (!isVs) { const ed = el("button", "ghost", S.rangeSel === r.id ? "Done" : "Settings"); ed.title = "Edit this range's own settings with the controls above"; ed.onclick = () => rangeEdit(r); row.appendChild(ed); }
      const go = el("button", "ghost", "▶"); go.title = "Go to this range"; go.onclick = () => gotoSrc(r.start);
      const x = el("button", "ghost", "✕"); x.title = "Remove this range"; x.onclick = () => rangeRemove(key, r);
      row.append(go, x); box.appendChild(row);
    }
    // From / To (only for Specific ranges: the whole episode needs no frames): the source frame numbers you type (as on the Timeline in frame mode); + Range adds them
    const fr = (ph, tip) => { const i = el("input", "mono"); i.type = "number"; i.min = "0"; i.step = "1"; i.placeholder = ph; i.title = tip; return i; };
    const inA = fr("from", "First frame of the range (source frame number)"), inB = fr("to", "Last frame of the range (source frame number)");
    const add = el("button", "ghost", "+ Range");
    const go = () => {
      const a = parseInt(inA.value, 10), b = parseInt(inB.value, 10);
      if (Number.isNaN(a) && Number.isNaN(b)) { rangeAdd(key); return; }        // nothing typed: 1/30 of the video at the playhead
      if (Number.isNaN(a) || Number.isNaN(b)) { status("Type both frame numbers: From and To", "err"); (Number.isNaN(a) ? inA : inB).focus(); return; }
      rangeAdd(key, Math.min(a, b), Math.max(a, b));
    };
    add.onclick = go;
    for (const i of [inA, inB]) i.onkeydown = (e) => { if (e.key === "Enter") go(); };
    add.title = "Add the range From - To (both frames included). Nothing typed: 1/30 of the video at the playhead";
    if (only) {
      const ff = el("div", "field rng-frames"), wrap = el("div", "rng-fr");
      wrap.append(inA, el("span", "muted", "to"), inB, add);
      ff.append(el("label", null, "Frames"), wrap); box.appendChild(ff);
    }
    g.classList.toggle("editing-range", !isVs && rangeList(key).some((r) => r.id === S.rangeSel));
  }
}
// timeline lanes: one per step / group that has ranges (drag to move, drag the ends to trim, click = edit its settings)
// A filter dragged (by its handle) onto the Timeline runs only there: the range starts where it is dropped and lasts 1/30 of the video
// (drag its ends to stretch it, drag it to move it - the lane below the layers). A filter that shares a group with
// others is split off into a group of its own first, so only it gets the range.
function initFilterDrop() {
  const body = $("ltBody");
  if (!body || body._drop) return;
  body._drop = true;
  const mime = "application/x-eva-vsf", stepPre = "application/x-eva-step-";
  const stepOf = (e) => { const t = [...e.dataTransfer.types].find((x) => x.startsWith(stepPre)); return t ? t.slice(stepPre.length) : null; };
  const NOT_PARTIAL = { telecine: "Telecine and interlacing works on the raw fields of the whole episode", crop: "Crop changes the picture's size for the whole episode", size: "Size changes the picture's size for the whole episode" };
  const COLOUR_NEW = "application/x-eva-clayer-new", COLOUR_LAYER = "application/x-eva-clayer", COLOUR_LUT = "application/x-eva-lut";
  const colourDrag = (e) => [COLOUR_NEW, COLOUR_LAYER, COLOUR_LUT].find((t) => e.dataTransfer.types.includes(t)) || null;
  const mark = el("div", "tlx-dropmark hidden"); body.appendChild(mark);
  const place = (e) => {
    const f = ltFrame(e.clientX), w = ltWindow(), len = rangeDefaultFrames();
    mark.style.left = ltX(f, w) + "%"; mark.style.width = Math.max(0.4, ltX(f + len, w) - ltX(f, w)) + "%";
    return f;
  };
  const hide = () => { mark.classList.add("hidden"); $("ltTip").classList.add("hidden"); };
  body.addEventListener("dragover", (e) => {
    const step = stepOf(e);
    if (!S.info || !([...e.dataTransfer.types].includes(mime) || step || colourDrag(e))) return;
    if (step && !RANGE_STEPS.includes(step)) { mark.classList.add("hidden"); e.dataTransfer.dropEffect = "none"; ltTip("can't run on a part of the episode", e.clientX); return; }   // no preventDefault = no drop
    e.preventDefault(); e.dataTransfer.dropEffect = "copy";
    const f = place(e); mark.classList.remove("hidden");
    ltTip(`${tc(f).slice(0, -4)} · runs here for ${rangeDefaultFrames()} frames (1/30 of the video)`, e.clientX);
  });
  body.addEventListener("dragleave", (e) => { if (!body.contains(e.relatedTarget)) hide(); });
  body.addEventListener("drop", (e) => {
    const step = stepOf(e);
    const data = e.dataTransfer.getData(mime);
    hide();
    if (!S.info) return;
    const cd = colourDrag(e);
    if (cd) {                                      // from the Colour tab: a new layer, a layer moved, or a LUT
      e.preventDefault();
      const f1 = ltFrame(e.clientX), ln2 = laneAtPoint(e.clientX, e.clientY), row = ln2 ? +ln2.dataset.layer : 0;
      if (cd === COLOUR_NEW) layerAtDrop(f1, row);
      else if (cd === COLOUR_LAYER) layerMoveTo(+e.dataTransfer.getData(cd), f1, row);
      else lutToTimeline(+e.dataTransfer.getData(cd), f1, row);
      return;
    }
    if (step) {                                    // a Filtering step: from the drop point for 1/30 of the video, and only there
      e.preventDefault();
      if (!RANGE_STEPS.includes(step)) { status((NOT_PARTIAL[step] || "This step") + ", so it cannot run on a part of it.", "err"); return; }
      const f0 = ltFrame(e.clientX);
      S.range_only = { ...(S.range_only || {}), [step]: true };
      const ln0 = laneAtPoint(e.clientX, e.clientY);
      rangeAdd(step, f0, null, ln0 ? +ln0.dataset.layer : 0);
      status(`${RANGE_NAME[step]} now runs only on this part (${rangeDefaultFrames()} frames = 1/30 of the video, from ${tc(f0).slice(0, -4)}); its controls set this part's own settings - stretch or move the bar on the Timeline. Click the bar again for the episode's own settings.`, "ok");
      return;
    }
    if (!data) return;
    e.preventDefault();
    const [gid, idx] = data.split("|");
    let g = S.vs_groups.find((x) => x.id === gid);
    if (!g) return;
    const f = ltFrame(e.clientX);
    if (g.filters.length > 1 && idx !== undefined && idx !== "") {
      const [filt] = g.filters.splice(+idx, 1);
      const ng = vsfNewGroup(vsfAfter(g), [filt]);
      S.vs_groups.splice(S.vs_groups.indexOf(g) + 1, 0, ng);
      g = ng;
    }
    const ln1 = laneAtPoint(e.clientX, e.clientY);
    rangeAdd("vs:" + g.id, f, null, ln1 ? +ln1.dataset.layer : 0);
    gotoSrc(f);
  });
}
function rangeLanes(heads, lanes, w) {
  const items = ensureLayers();
  const N = Math.max(0, ...items.map((x) => x.r.layer));
  for (let L = 1; L <= N; L++) {
    const th = el("div", "th rg"); th.title = "Drag a range up or down to put it on another layer";
    const del = el("button", "i", "✕"); del.title = `Delete Layer ${L} and the ranges on it`;
    del.onclick = (e) => { e.stopPropagation(); rangesDelete(items.filter((x) => x.r.layer === L)); };
    th.append(el("span", null, `Layer ${L}`), del); heads.appendChild(th);
    const lane = el("div", "lt-lane rg-lane"); lane.dataset.layer = L;
    for (const { key, r } of items) if (r.layer === L) { const bar = rangeBar(key, r, w); if (bar) lane.appendChild(bar); }
    lanes.appendChild(lane);
  }
}
let RSEL = null;                                   // the range picked on the Timeline: {key, id} (Delete key / right-click menu delete it)
function rangesDelete(items) {                     // [{key, r}] -> gone; says which step / filter is left without a range
  if (!items.length) return;
  const names = items.map(({ key, r }) => rangeTitle(key, r)), keys = new Set();
  for (const { key, r } of items) {
    if (key.startsWith("vs:")) { const g = S.vs_groups.find((x) => x.id === key.slice(3)); if (g) g.ranges = (g.ranges || []).filter((x) => x !== r); }
    else S.ranges = (S.ranges || []).filter((x) => x !== r);
    if (S.rangeSel === r.id) S.rangeSel = null;
    if (RSEL && RSEL.id === r.id) RSEL = null;
    keys.add(key);
  }
  if ([...keys].some((k) => k.startsWith("vs:"))) vsgChanged();
  if ([...keys].some((k) => !k.startsWith("vs:"))) changed("ranges", S.ranges);
  renderRangeBoxes(); renderLayerTrack(); sync();
  const bare = [...keys].filter((k) => !rangeList(k).length).map(rangeBase);
  status(`${names.length === 1 ? names[0] : names.length + " ranges"} deleted` + (bare.length ? ` - ${bare.join(", ")} ${bare.length > 1 ? "have" : "has"} no range left and runs nowhere (set it to Whole episode, or add a range)` : ""), "ok");
}
function rangeMenu(e, key, r) {                    // right-click on a bar
  e.preventDefault(); e.stopPropagation();
  closeRangeMenu();
  const m = el("div", "ctx-menu"); m.id = "rngMenu";
  const add = (t, fn) => { const b = el("button", null, t); b.onclick = () => { closeRangeMenu(); fn(); }; m.appendChild(b); };
  add(`Delete ${rangeTitle(key, r)}`, () => rangesDelete([{ key, r }]));
  add(`Delete ${rowLabel(r.layer)}`, () => removeTrack(r.layer));
  add("Go to this range", () => gotoSrc(r.start));
  document.body.appendChild(m);
  m.style.left = Math.min(e.clientX, innerWidth - m.offsetWidth - 6) + "px"; m.style.top = Math.min(e.clientY, innerHeight - m.offsetHeight - 6) + "px";
  setTimeout(() => document.addEventListener("pointerdown", closeOnOutside, true), 0);
}
function closeOnOutside(e) { if (!e.target.closest || !e.target.closest("#rngMenu")) closeRangeMenu(); }
function closeRangeMenu() { const m = document.getElementById("rngMenu"); if (m) m.remove(); document.removeEventListener("pointerdown", closeOnOutside, true); }
// Ctrl+C / Ctrl+X / Ctrl+V on a picked range: the copy has the same length, settings and filter, starts at the playhead and goes on
// the layer of the original when it fits there, else the next one where it does.
let RCLIP = null;                                  // {key, len, settings, layer}
function rangeCopy(it, cut) {
  RCLIP = { key: it.key, len: it.r.end - it.r.start, settings: structuredClone(it.r.settings || {}), layer: it.r.layer, name: rangeTitle(it.key, it.r) }; CLIPKIND = "range";
  if (cut) rangesDelete([it]);
  else status(`Copied ${RCLIP.name} (${RCLIP.len} frames) - Ctrl+V pastes it at the playhead`, "ok");
}
function rangePaste() {
  if (!RCLIP || !S.info) return;
  const { key } = RCLIP;
  const isVs = key.startsWith("vs:"), g = isVs ? S.vs_groups.find((x) => x.id === key.slice(3)) : null;
  if (isVs && !g) { status(`${RCLIP.name} cannot be pasted: its filter is gone.`, "err"); return; }
  const [e0, e1] = S.episodes[S.current];
  const len = Math.min(RCLIP.len, e1 - e0);
  const start = Math.max(e0, Math.min(playSrc(), e1 - len));
  const r = { id: "r" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5), start, end: start + len, on: true, settings: structuredClone(RCLIP.settings) };
  r.n = Math.max(0, ...rangeList(key).map((x, i) => x.n || i + 1)) + 1;
  const items = activeRangeItems(), N = Math.max(0, ...items.map((i) => i.r.layer));
  let L = Math.max(1, RCLIP.layer || 1);
  while (L <= N && items.some((i) => i.r.layer === L && rOverlap(i.r, r))) L++;
  r.layer = L;
  if (isVs) { g.ranges = [...(g.ranges || []), r]; g.only = true; }
  else { r.step = key; S.ranges = [...(S.ranges || []), r]; S.range_only = { ...(S.range_only || {}), [key]: true }; }
  RSEL = { key, id: r.id }; RPIN = r;
  rangesChanged(key);
  status(`Pasted ${rangeTitle(key, r)}: ${rangeLabel(r)} on Layer ${r.layer}`, "ok");
}
function initRangeSelection() {
  document.addEventListener("pointerdown", (e) => {   // a click anywhere else lets go of the picked range
    if (RSEL && !(e.target.closest && e.target.closest(".rg-blk, #rngMenu"))) { RSEL = null; renderLayerTrack(); }
  }, true);
  document.addEventListener("keydown", (e) => {       // Ctrl+C / X / V on the picked range
    if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey) return;
    const k = e.key.toLowerCase();
    if (k !== "c" && k !== "x" && k !== "v") return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) || String(getSelection()).length) return;   // ordinary text copy / paste stays
    if (k === "v") {
      if (CLIPKIND === "layer" ? !LCLIP : !RCLIP) return;
      e.preventDefault(); e.stopImmediatePropagation(); CLIPKIND === "layer" ? layerPaste() : rangePaste(); return;
    }
    const it = RSEL && activeRangeItems().find((i) => i.r.id === RSEL.id);
    if (it) { e.preventDefault(); e.stopImmediatePropagation(); rangeCopy(it, k === "x"); return; }
    if ((TLX.focus === "layer" || S.step === "colour") && S.layers[S.layerSel]) { e.preventDefault(); e.stopImmediatePropagation(); layerCopy(S.layerSel, k === "x"); }
  }, true);
  document.addEventListener("keydown", (e) => {       // Delete removes the picked range (before the Timeline's own Delete = cut a part)
    if (!RSEL || (e.key !== "Delete" && e.key !== "Backspace")) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    const it = activeRangeItems().find((i) => i.r.id === RSEL.id);
    if (!it) { RSEL = null; return; }
    e.preventDefault(); e.stopImmediatePropagation();
    rangesDelete([it]);
  }, true);
}
const laneAtPoint = (x, y) => { const e = document.elementsFromPoint(x, y).find((n) => n.classList && n.classList.contains("rg-lane")); return e || null; };
function rangeBar(key, r, w) {                     // one range on the Timeline: its filter's colour, "Compression R1", drag / stretch / change layer
  const isVs = key.startsWith("vs:");
  const x0 = Math.max(-1, ltX(r.start, w)), x1 = Math.min(101, ltX(r.end, w));
  if (x1 <= 0 || x0 >= 100) return null;
  const bar = el("div", "lt-layer rg-blk" + (S.rangeSel === r.id ? " sel" : "") + (RSEL && RSEL.id === r.id ? " rsel" : ""), rangeTitle(key, r));
  bar.addEventListener("contextmenu", (e) => rangeMenu(e, key, r));
  bar.style.left = x0 + "%"; bar.style.width = Math.max(0.3, x1 - x0) + "%"; bar.style.background = rangeColor(key);
  bar.title = `${rangeTitle(key, r)}: ${rangeLabel(r)} (${tc(r.start).slice(0, -4)}) · drag to move anywhere (up / down = another layer, above / below the layers = a new one), drag the ends to trim · click to pick it (Delete removes it)${isVs ? "" : ", click again for its settings"} · Ctrl+C / Ctrl+V copies it to the playhead · right-click: delete range / layer`;
  bar.addEventListener("pointerdown", (e) => {
    e.stopPropagation(); e.preventDefault();
    const br = bar.getBoundingClientRect();
    const ew = Math.min(7, br.width / 3);          // the grab zones at the ends shrink with a narrow bar (zoom in to stretch a short one)
    const mode = e.clientX - br.left < ew ? "a" : br.right - e.clientX < ew ? "b" : "move";
    const f0 = ltFrame(e.clientX), s0 = r.start, e0 = r.end; let moved = false, newLayer = null;
    bar.setPointerCapture(e.pointerId);
    const mv = (ev) => {
      const d = ltFrame(ev.clientX) - f0;
      if (Math.abs(ev.clientX - e.clientX) > 3 || Math.abs(ev.clientY - e.clientY) > 3) moved = true;
      if (!moved) return;
      const [a, b] = S.episodes[S.current];
      if (mode === "move") { const len = e0 - s0; r.start = Math.max(a, Math.min(b - len, s0 + d)); r.end = r.start + len; }
      else if (mode === "a") r.start = Math.max(a, Math.min(r.end - 1, s0 + d));
      else r.end = Math.min(b, Math.max(r.start + 1, e0 + d));
      bar.style.left = ltX(r.start, w) + "%"; bar.style.width = Math.max(0.3, ltX(r.end, w) - ltX(r.start, w)) + "%";
      const ln = mode === "move" ? laneAtPoint(ev.clientX, ev.clientY) : null;
      const all = $$(".rg-lane");
      const line = $("rgDropLine") || (() => { const l = el("div", "rg-dropline hidden"); l.id = "rgDropLine"; $("ltBody").appendChild(l); return l; })();
      let tip = `${rangeLabel(r)} · Layer ${r.layer}`;
      newLayer = null; line.classList.add("hidden");
      if (ln && +ln.dataset.layer !== r.layer) {     // over another layer: it goes there (what it overlaps is pushed to another layer when you let go)
        r.layer = +ln.dataset.layer; ln.appendChild(bar);
        tip = `${rangeLabel(r)} · Layer ${r.layer}`;
      } else if (!ln && mode === "move" && all.length) {   // above the first layer / below the last one: a new layer there
        const top = all[0].getBoundingClientRect(), bot = all[all.length - 1].getBoundingClientRect(), bb = $("ltBody").getBoundingClientRect();
        if (ev.clientY < top.top - 2) newLayer = "above"; else if (ev.clientY > bot.bottom + 2) newLayer = "below";
        if (newLayer) {
          line.style.top = ((newLayer === "above" ? top.top : bot.bottom) - bb.top - 1) + "px"; line.classList.remove("hidden");
          tip = `${rangeLabel(r)} · new layer ${newLayer}`;
        }
      }
      ltTip(tip, ev.clientX);
    };
    const up = () => {
      bar.removeEventListener("pointermove", mv); $("ltTip").classList.add("hidden");
      const dl = $("rgDropLine"); if (dl) dl.classList.add("hidden");
      if (newLayer) r.layer = newLayer === "above" ? S.tracks + 0.5 : 0.5;   // settleRows renumbers
      if (moved) RPIN = r;
      RSEL = { key, id: r.id };
      if (moved) rangesChanged(key); else if (!isVs) rangeEdit(r); else renderLayerTrack();
    };
    bar.addEventListener("pointermove", mv); bar.addEventListener("pointerup", up, { once: true });
  });
  return bar;
}

// ---------------------------------------------------------------- A / B snapshots and bookmarks
function scopesShown() { const g = document.querySelector('[data-fold="scopes"]'); return document.body.classList.contains("step-colour") && !(g && g.classList.contains("closed")); }   // a folded Scopes window costs nothing   // the scopes: Colour Inspector only
function snapState(sn) {                           // the backend state of the episode shown, with a snapshot's settings
  const live = pictureSnap();
  pictureLoad(sn.pic);
  const st = backendState([S.current]);
  pictureLoad(live);
  return st;
}
function vsviewState() {
  const st = backendState([S.current]);
  st.snap_states = (S.snaps || []).map((sn) => ({ name: sn.name, state: snapState(sn) }));
  return st;
}
function renderAbSel() {
  const sel = $("abSel");
  sel.innerHTML = "";
  for (const sn of S.snaps || []) sel.appendChild(el("option", null, sn.name)).value = sn.id;
  if (S.abSnap) sel.value = S.abSnap;
}
function snapshotTake() {
  if (!S.info) return;
  S.snaps = S.snaps || [];
  const id = "s" + Date.now().toString(36);
  const name = `${S.snaps.length + 1} · ${new Date().toTimeString().slice(0, 5)}`;
  S.snaps.push({ id, name, pic: pictureSnap() });
  S.abSnap = id; renderAbSel();
  log(`Snapshot ${name} kept - change settings, then A / B compares them with it (also a vsview tab).`, "ok");
  status(`Snapshot ${name} kept. Change something, then A / B shows now vs. the snapshot.`, "ok");
  vsviewFollow();
}
async function gotoSrc(f) {
  try {
    const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: f });
    S.frame = r.frame; S.total = Math.max(S.total, r.frame + 1); schedulePreview(30);
  } catch { /* logged */ }
}
function bookmarkAdd() {
  if (!S.info) return;
  const f = S.src != null ? S.src : Math.round(headSrcFrame());
  S.bookmarks = S.bookmarks || [];
  if (S.bookmarks.some((b) => Math.abs(b.src - f) < 2)) return;
  S.bookmarks.push({ src: f, note: `★ ${tc(f).slice(0, -4)}` });
  S.bookmarks.sort((a, b) => a.src - b.src);
  renderTimeline(); status(`Bookmarked ${tc(f).slice(0, -4)} - click its ★ on the timeline to come back.`, "ok");
}

// ---------------------------------------------------------------- Media: several open sources, each with its whole state
const SOURCES = [];                                // [{path, name, state}] - the current one's state lives in S
const SRC_KEYS = ["info", "detected", "episodes", "current", "out_dir", "epSet", "frame", "total", "src", "script", "bookmarks", "snaps"];
function sourceStash() {
  if (!S.info) return;
  epSetAlign(); S.epSet[S.current] = pictureSnap();
  const st = {};
  for (const k of SRC_KEYS) st[k] = structuredClone(S[k] ?? null);
  st.checked = [...S.checked];
  let it = SOURCES.find((x) => x.path === S.info.path);
  if (!it) { it = { path: S.info.path, name: basename(S.info.path) }; SOURCES.push(it); }
  it.state = st;
}
function sourceRegister() {
  if (!S.info || SOURCES.some((x) => x.path === S.info.path)) { renderSources(); return; }
  SOURCES.push({ path: S.info.path, name: basename(S.info.path), state: null });
  renderSources();
}
function switchSource(path) {
  if (S.info && S.info.path === path) return;
  const it = SOURCES.find((x) => x.path === path);
  if (!it || !it.state) return;
  togglePlay(false); histCommit();
  sourceStash();
  const st = structuredClone(it.state);
  for (const k of SRC_KEYS) S[k] = st[k];
  S.checked = new Set(st.checked || []);
  S.bookmarks = S.bookmarks || []; S.snaps = S.snaps || [];
  S.vfrAuto = null; S.queue = {};
  pictureLoad(S.epSet[S.current] || pictureFresh());
  $("discBox").classList.add("hidden"); $("dropZone").classList.add("hidden");
  $("srcCard").classList.remove("hidden"); $("placeholder").classList.add("hidden");
  showSourceInfo(); renderEpisodes(); sync(); renderSources(); refreshScript(true); renderAbSel();
  if (S.view === "ab" && !(S.snaps || []).length) setView("sbs");
  log(`Switched to ${it.name}`, "ok"); status(`${it.name} - its own episodes and settings.`);
  schedulePreview(30);
}
function closeSource(path) {
  const i = SOURCES.findIndex((x) => x.path === path);
  if (i < 0) return;
  if (S.info && S.info.path === path) {
    const other = SOURCES.find((x) => x.path !== path && x.state);
    if (!other) { status("This is the only open source - open another one first, or just open a new file over it."); return; }
    switchSource(other.path);
  }
  SOURCES.splice(SOURCES.findIndex((x) => x.path === path), 1);
  renderSources();
}
function renderSources() {
  const box = $("srcList");
  box.innerHTML = "";
  if (SOURCES.length < 2) return;                    // one source: the Source card below says it all
  box.appendChild(el("div", "hd", `${SOURCES.length} open sources`));
  for (const it of SOURCES) {
    const cur = S.info && S.info.path === it.path;
    const st = cur ? { info: S.info, episodes: S.episodes, checked: [...S.checked] } : it.state || {};
    const i = st.info || {};
    const kind = i.image ? "image" : i.progressive ? "progressive" : i.pal ? "PAL" : "NTSC";
    const row = el("div", "src-item" + (cur ? " cur" : ""));
    row.title = it.path;
    const x = el("button", null, "✕"); x.title = "Close this source (its settings are kept until the app closes)";
    x.onclick = (e) => { e.stopPropagation(); closeSource(it.path); };
    row.append(el("b", null, it.name), x,
               el("small", null, `${kind} · ${(st.episodes || []).length} episode(s) · ${(st.checked || []).length} ticked`));
    row.onclick = () => switchSource(it.path);
    box.appendChild(row);
  }
}

// ---------------------------------------------------------------- presets (per show; presets\ via the backend)
let PRESETS = [];
function presetWorkflow() {
  const settings = {};
  for (const [k, v] of Object.entries(pictureSnap())) if (!WF_SKIP.has(k)) settings[k] = v;
  return { app: "EVA toolkit", kind: "workflow", version: 1, saved_with: document.querySelector(".brand .ver")?.textContent || "",
           created: new Date().toISOString(), source: S.info ? basename(S.info.path) : "", settings };
}
async function renderPresets() {
  try { PRESETS = await rpc("presets", {}); } catch { PRESETS = []; }
  const box = $("presetList");
  box.innerHTML = "";
  if (!PRESETS.length) box.appendChild(el("div", "note", "No presets yet. Set up an episode, then save its settings here by name."));
  for (const p of PRESETS) {
    const row = el("div", "preset-item");
    const ap = el("button", null, "Apply"); ap.disabled = !S.info; ap.onclick = () => presetApply(p.name);
    const del = el("button", "ghost", "✕"); del.title = "Delete this preset";
    del.onclick = async () => { if (!confirm(`Delete the preset "${p.name}"?`)) return; await rpc("preset_delete", { name: p.name }); renderPresets(); };
    row.append(el("b", null, p.name), ap, del, el("small", null, `${p.keys} settings${p.source ? " · from " + p.source : ""}${p.created ? " · " + p.created.slice(0, 10) : ""}`));
    box.appendChild(row);
  }
  const sel = $("presetSel"), cur = sel.value;
  sel.innerHTML = "";
  sel.appendChild(el("option", null, "-")).value = "";
  for (const p of PRESETS) sel.appendChild(el("option", null, p.name)).value = p.name;
  sel.value = PRESETS.some((p) => p.name === cur) ? cur : "";
}
function presetSave() {
  if (!S.info) { status("Open a source first: a preset is the settings of the episode shown.", "err"); return; }
  showLeft("presets");
  setTimeout(() => $("presetName").focus(), 50);
}
async function presetApply(name) {
  if (!S.info) { status("Open a source first.", "err"); return; }
  try { applyWorkflow(await rpc("preset_get", { name }), `preset "${name}"`); }
  catch (e) { status(`Preset not applied: ${e.message || e}`, "err"); }
}
function initPresets() {
  $("presetForm").onsubmit = async (e) => {
    e.preventDefault();
    const name = $("presetName").value.trim();
    if (!name) { $("presetName").focus(); return; }
    if (!S.info) { status("Open a source first: a preset is the settings of the episode shown.", "err"); return; }
    if (PRESETS.some((p) => p.name.toLowerCase() === name.toLowerCase()) && !confirm(`Replace the preset "${name}"?`)) return;
    histCommit();
    await rpc("preset_save", { name, workflow: presetWorkflow() });
    $("presetName").value = "";
    log(`Preset saved: ${name}`, "ok"); status(`Preset "${name}" saved - apply it to any episode or source of the show.`, "ok");
    renderPresets();
  };
  $("presetSel").onchange = () => { const n = $("presetSel").value; if (n) presetApply(n); };
  renderPresets();
}

// ---------------------------------------------------------------- updates (app/update.js in the main process)
const UPD = { info: null, state: "idle", hidden: false };
function updBanner() {
  const b = $("updBanner"), i = UPD.info;
  b.classList.toggle("hidden", !i || !i.newer || UPD.hidden);
  if (!i || !i.newer) return;
  const mb = i.asset ? Math.round(i.asset.size / 1048576) + " MB" : "";
  $("updTitle").textContent = `${i.name || i.latest} is available`;
  $("updText").textContent = UPD.state === "downloading" ? "Downloading…" : UPD.state === "unpack" ? "Unpacking…"
    : UPD.state === "ready" ? "Downloaded. Restart to install it (your settings and engine stay)."
    : UPD.state === "error" ? UPD.error : `You have ${i.current}.${mb ? " Download " + mb + "." : ""}`;
  $("updBar").classList.toggle("hidden", !["downloading", "unpack"].includes(UPD.state));
  $("btnUpdGo").textContent = UPD.state === "ready" ? "Restart & install" : UPD.state === "downloading" || UPD.state === "unpack" ? "Cancel" : "Update";
  $("btnUpdLater").classList.toggle("hidden", UPD.state === "downloading" || UPD.state === "unpack");
}
async function updCheck(manual) {
  if (manual) $("updStatus").textContent = "Checking…";
  const r = await window.api.updateCheck();
  if (r.error) {
    if (manual) $("updStatus").textContent = `Could not check: ${r.error}`;
    return;
  }
  UPD.info = r; UPD.hidden = false;
  $("updStatus").textContent = r.newer ? `${r.latest} is available (you have ${r.current}).` : `You have the latest version (${r.current}).`;
  if (r.newer) log(`Update available: ${r.name || r.latest} (you have ${r.current})`, "ok");
  updBanner();
}
async function updGo() {
  if (UPD.state === "downloading" || UPD.state === "unpack") { await window.api.updateCancel(); return; }
  if (UPD.state === "ready") {
    if (S.busy || QUEUE.running) { status("Finish (or stop) the export first - installing restarts the app.", "err"); return; }
    if (!confirm(`Install ${UPD.info.latest} now?\n\nEVA toolkit closes, the new files are copied over this folder and the new version starts. ` +
                 "Your settings, the installed engine, engines\\, temp\\ and your own filters stay.")) return;
    const r = await window.api.updateInstall();
    if (r.error) { UPD.state = "error"; UPD.error = r.error; updBanner(); }
    return;
  }
  UPD.state = "downloading"; updBanner();
  const r = await window.api.updateDownload();
  if (r.error) { UPD.state = r.error === "cancelled" ? "idle" : "error"; UPD.error = `Download failed: ${r.error}`; }
  else { UPD.state = "ready"; log(`Update ${UPD.info.latest} downloaded - restart to install it.`, "ok"); }
  updBanner();
}
function initUpdates() {
  $("btnUpdGo").onclick = updGo;
  $("btnUpdNotes").onclick = () => window.api.openRelease();
  $("btnUpdLater").onclick = () => { UPD.hidden = true; updBanner(); };
  $("btnUpdCheck").onclick = () => updCheck(true);
  window.api.onUpdate((m) => {
    if (m.stage === "download") { UPD.state = "downloading"; $("updFill").style.width = (m.total ? 100 * m.got / m.total : 0).toFixed(1) + "%"; }
    else if (m.stage === "unpack") { UPD.state = "unpack"; $("updFill").style.width = "100%"; }
    updBanner();
  });
  if (S.auto_update !== false && !window.__selftest) setTimeout(() => updCheck(false).catch(() => {}), 4000);
}

async function init() {
  bind();
  let hello;
  for (let i = 0; i < 50; i++) {                       // backend may still be starting
    try { hello = await rpc("hello", {}); break; } catch { await new Promise((r) => setTimeout(r, 200)); }
  }
  if (!hello) { status("Could not start the Python backend.", "err"); return; }
  S.opts = hello.options;
  if (hello.busy) { S.busy = true; log("A task is still running in the engine (the window was reloaded): Stop ends it.", "warn"); }   // the Stop button must stay reachable
  for (const k of SAVED) if (hello.settings && k in hello.settings) S[k] = hello.settings[k];
  if (!S.opts.ai_model.some((o) => o[0] === S.ai_model)) S.ai_model = "precise";   // UniNet / StableNet were removed from the menu
  buildControls(); buildAdvanced(); initColourUi(); initTempUi(); initWorkspace(); goto("source"); sync();

  initUpdates();
  if (hello.missing.length) { log("Missing tools: " + hello.missing.join(", "), "err"); status("Missing tools: " + hello.missing.join(", "), "err"); }
  else status("Ready - open or drop a DVD or video.");
  autosaveMark();
  let again = null;
  try { again = JSON.parse(localStorage.getItem("reopenAfterReload") || "null"); localStorage.removeItem("reopenAfterReload"); } catch { /* none */ }
  if (again && again.path) {                           // a reload keeps your place: the same source, and its settings come back with it
    try { await loadVideo(again.path); if (again.current > 0 && again.current < S.episodes.length) selectEpisode(again.current); } catch { /* shown by loadVideo */ }
  }
}
init();
