// DVD Restore - renderer. All processing happens in the Python backend; this file is UI + state.
// Flow: Source (open / import disc / episodes) -> Analyze (all detectors) -> Repair (pipeline order, one Auto per group) -> Colour -> Crop & size -> Script -> Export.
"use strict";
const $ = (id) => document.getElementById(id);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const S = {
  opts: null, info: null, episodes: [], detected: [], checked: new Set(), current: 0,
  step: "source", analyzed: false, auto: {}, disc: null, titles: [], titleSel: 0, queue: {},
  mode: "vfr", order: "tff", pal_mode: "auto", adv: {}, comb: "nnedi3", ai_model: "precise", stab: "off", stab_level: "normal",
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
  levels: null, srcDir: "",
  aspect: "auto", aspect_custom: "", custom_w: 960, custom_h: 720, custom_lock: true,
  size: "keep", kernel: "spline36", kernel_b: 0.33, kernel_c: 0.33, tfix_strength: 2.0, encoder: "x264", crf: 16, out_dir: "", name: "Episode_{n:02d}",
  enc_rate: "q", enc_bitrate: 8000, enc_preset: "", enc_tune: "", enc_depth: 10, enc_chroma: "420", enc_keyint: 0, enc_extra: "",
  audio_codec: "copy", audio_bitrate: 192,
  busy: false, view: "after", zoom: "Fit", split: 50, img: null, frame: null, total: 0,
  backend: "pytorch", env: null,
  sharp: { luma: 0, chroma: 0, mode: "faithful", radius: "fine", lines: 0 },
  upscale: { models: [] },              // [{path, name, info, strength, on, engine: {...}}]
  script: null, scriptGen: "",          // script = the user's edited text (null = follow the settings)
};
// Preferences: kept between sessions and for every DVD. Picture settings (PICTURE below) are not: they belong to
// one episode of one source and start from the defaults for every new DVD.
const SAVED = ["temp_dir", "stage_order", "backend", "ai_model", "encoder", "crf", "name", "adv",
               "enc_rate", "enc_bitrate", "enc_preset", "enc_tune", "enc_depth", "enc_chroma", "enc_keyint", "enc_extra", "audio_codec", "audio_bitrate"];
// Picture settings: Repair, Colour (with its adjustment layers), Crop & size, and what Auto measured.
// Every episode has its own (S.epSet[k]); a new episode starts from the defaults.
const PICTURE = ["mode", "order", "pal_mode", "comb", "ivtc_repair", "chroma_unmix", "aa", "deblend", "crawl", "rainbow", "composite", "rainbow_fill",
                 "deblock", "ghost", "ghost_strength", "ghost_taps", "halo_rx", "halo_method", "sx", "sy", "bleed", "warp", "neutral",
                 "stab", "stab_level", "tfix_strength", "crop", "aspect", "aspect_custom", "size", "custom_w", "custom_h", "custom_lock",
                 "uni_crawl", "uni_rainbow", "uni_mpeg", "uni_halo", "uni_chroma", "uni_grain", "uni_detail",
                 "kernel", "kernel_b", "kernel_c", "sharp", "upscale", "gradeAll", "layers", "tracks", "trackOff", "auto", "levels"];
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
  if (S.info) { o.order = S.info.order; o.pal_mode = "auto"; }
  return o;
}
function epSetAlign() { while (S.epSet.length < S.episodes.length) S.epSet.push(null); S.epSet.length = S.episodes.length; }
const NO_PREVIEW = new Set(["mode", "encoder", "crf", "out_dir", "name", "enc_rate", "enc_bitrate", "enc_preset", "enc_tune",
                            "enc_keyint", "enc_extra", "audio_codec", "audio_bitrate"]);
const NO_SCRIPT = new Set(["out_dir", "name"]);   // settings the preview does not show

// ---------------------------------------------------------------- helpers
const fps = () => (S.info ? S.info.fps_num / S.info.fps_den : 30000 / 1001);
function tc(frames) {
  const s = frames / fps();
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
  return Math.round(s * fps());
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
    info: S.info, episodes: S.episodes, selected,
    mode: S.mode, order: S.order, pal_mode: S.pal_mode, adv: S.adv, comb: S.comb, ai_model: S.ai_model, stab: S.stab, stab_level: S.stab_level, tfix_strength: S.tfix_strength,
    uni_crawl: S.uni_crawl, uni_rainbow: S.uni_rainbow, uni_mpeg: S.uni_mpeg, uni_halo: S.uni_halo, uni_chroma: S.uni_chroma, uni_grain: S.uni_grain, uni_detail: S.uni_detail,
    crawl: S.crawl, rainbow: S.rainbow, composite: S.composite,
    ghost: S.ghost, ghost_strength: S.ghost_strength, ghost_taps: S.ghost_taps, halo_rx: S.halo_rx, halo_method: S.halo_method, deblock: S.deblock,
    ivtc_repair: S.ivtc_repair, chroma_unmix: S.chroma_unmix, deblend: S.deblend, aa: S.aa, rainbow_fill: S.rainbow_fill,
    sx: S.sx, sy: S.sy, bleed: S.bleed, warp: S.warp, neutral: S.neutral,
    crop: S.crop, size: S.size, kernel: S.kernel, kernel_b: S.kernel_b, kernel_c: S.kernel_c, encoder: S.encoder, crf: S.crf,
    enc_rate: S.enc_rate, enc_bitrate: S.enc_bitrate, enc_preset: S.enc_preset, enc_tune: S.enc_tune, enc_depth: S.enc_depth,
    enc_chroma: S.enc_chroma, enc_keyint: S.enc_keyint, enc_extra: S.enc_extra, audio_codec: S.audio_codec, audio_bitrate: S.audio_bitrate,
    grade: S.gradeAll, layers: [...S.layers].sort((p, q) => (p.track || 1) - (q.track || 1) || p.start - q.start)     // bottom track first
      .map((l) => ({ start: l.start, end: l.end, on: l.on !== false && !S.trackOff[l.track || 1], opacity: l.opacity ?? 1, grade: l.grade })),
    stage_order: S.stage_order,
    aspect: S.aspect, aspect_custom: S.aspect_custom,
    custom_w: S.custom_w, custom_h: S.custom_h, custom_lock: S.custom_lock,
    out_dir: S.out_dir, name: S.name, temp_dir: S.temp_dir || "", backend: S.backend, sharp: S.sharp, upscale: S.upscale,
    script: S.script,
  };
}
function saveSettings() {
  if (window.__selftest) return;                 // automated UI checks must never change the user's settings
  const s = {};
  for (const k of SAVED) s[k] = S[k];
  window.api.rpc("save_settings", { settings: s });
}

// ---------------------------------------------------------------- generic bound controls
// data-bind="key" (or "crop.l") on a select, .seg, .stepper, checkbox or text input.
function getv(key) { return key.split(".").reduce((o, k) => o[k], S); }
function setv(key, v) { const ks = key.split("."), last = ks.pop(); ks.reduce((o, k) => o[k], S)[last] = v; }

function changed(key, v) {
  if (PLAY.on && !NO_PREVIEW.has(key)) togglePlay(false);
  setv(key, v);
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
function sync() {
  const pal = !!(S.info && S.info.pal);
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
  $("aiNote").textContent = {
    field: "FieldNet: one network for the whole cleanup - its own IVTC repair from the raw fields, dot crawl, rainbows, MPEG blocking, halos and colour bleed, recurrent over time. After it the picture is steadied (motion-compensated averaging, fine-detail and still-area smoothing, and a long still-area lock), so shimmer, flicker and wobble on still detail such as door mesh or rug lettering go away without losing sharpness. Each dial: 1 = remove fully, 0 = leave as on the disc; while a dial is above 0 the matching classic step below is left out. Grain keep 1 = film grain and noise stay. Detail recovery 1 = fine detail an old decoder cut out is rebuilt.",
    uni: "All-in-one network: dot crawl, rainbows, MPEG blocking / flicker, halos and colour bleed / shift in one pass, recurrent over time, so it flickers less than the other cleaners - for fully steady still detail (door mesh, rug lettering) add the Steady chain stabilizer. Each dial: 1 = remove fully, 0 = leave as on the disc. While a dial is above 0 the matching classic step (MPEG repair, halo removal, colour shift / bleed fix) is left out. Grain keep 1 = film grain and noise stay. Compared with StableNet it is a little softer on the finest detail.",
    stable: "StableNet: recurrent network for dots, rainbows and MPEG flicker; steady by itself, no stabilizer needed. It also removes film grain.",
    temporal: "Neural network that looks at 5 frames: dots and rainbows invert from frame to frame, real detail does not. With it on, the classic luma filters are left out (they cost detail); the classic rainbow filter stays.",
    xnet: "XNet: the 5-frame network trained for steadiness (no wobble on fine texture) and to leave grain and detail alone; its luma correction can only act where dot crawl can exist.",
    precise: "Single-frame version of the same network (use if the temporal one shows ghosting on a source).",
    off: "Only the classic temporal filters (they soften fine detail - the AI is better).",
  }[S.ai_model] || "";
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
  $("levelNote").textContent = L ? `Measured: darkest picture tone ${L.level_black}, brightest ${L.level_white} (DVD black = 16, white = 235).` : "Press Auto colour to measure the black and white level of this source.";
  $("gamutNote").textContent = (L ? `Measured: ${L.gamut_out} % of pixels are stored stronger than a screen can show, ${L.clip_flat} % sit flat at the limit (worst tenth of frames). ` : "") + ({
    off: "", compress: "Compress: uses the colour the disc really stores beyond the range. Right for DVDs.",
    ai: "AI rebuild: for colour that was cut off before the disc or capture was made. It estimates the lost gradation from the other colour channels and the surroundings; where all three channels are flat there is nothing to go on.",
  }[S.grade.gamut] || "");
  $("wbNote").textContent = !L ? "Auto white balance measures the colour of the whites (paper, eyes, highlights) and makes them neutral."
    : L.wb_u == null ? `Too few white areas in this source to judge the white balance (${L.wb_frames} frames with whites).`
    : `Measured: whites sit at Cb ${sign(L.wb_u)} / Cr ${sign(L.wb_v)} levels from grey, the same in ${L.wb_consistent} % of the frames with whites` +
      ((L.grade.temp || L.grade.tint) ? ` - a cast from the transfer: corrected with temperature ${L.grade.temp}, tint ${L.grade.tint}.`
        : (Math.hypot(L.wb_u, L.wb_v) < 1.5 ? " - neutral, nothing to correct." : " - changes from shot to shot, so it is the art (warm / cold scenes), left alone."));
  const trt = S.backend === "tensorrt";
  $("backendBadge").textContent = trt ? "TensorRT" : "PyTorch"; $("backendBadge").classList.toggle("trt", trt);
  $("backendNote").textContent = trt ? "Each model is compiled for this GPU the first time it meets a frame size (5-25 s each, then cached). The status line says when it happens."
                                     : "Models run directly in PyTorch.";
  $("exportNote").textContent = S.script ? "Export uses your edited script (Script step) for every ticked episode." : "";
  $("scriptFlag").classList.toggle("hidden", !S.script);
  $("srcShort").textContent = S.info ? basename(S.info.path) : "no source";
  $("sharpNote").textContent = (S.sharp.luma > 0 || S.sharp.chroma > 0)
    ? (S.sharp.mode === "faithful" ? "Faithful: edges get steeper but never overshoot the pixels around them, so no halos." : "Unsharp: crisper, but can add bright / dark rims along outlines - check Split.")
    : "Applied after resizing, so it acts on the final picture.";
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
  $("btnExport").textContent = S.checked.size ? `Export ${S.checked.size} episode${S.checked.size === 1 ? "" : "s"}` : "Export";
  $("btnShowOut").disabled = !S.out_dir;
  $("nextAnalyze").disabled = !has;
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
  for (const s of stageOrder()) nav.insertBefore(nav.querySelector(`[data-step="${STAGE_STEP[s]}"]`), anchor);
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
  if (step !== "source" && !S.info) return;
  S.step = step;
  document.body.classList.toggle("step-script", step === "script");
  if (step === "script") refreshScript();
  if (step === "colour") requestScopes();
  setTimeout(layoutImage, 0);
  for (const p of $$(".panel")) p.classList.toggle("hidden", p.dataset.panel !== step);
  renderLayerTrack();
  if (step === "analyze") $("anEpisode").textContent = `episode ${S.current + 1}` + (S.episodes.length > 1 ? ` of ${S.episodes.length}` : "");
  updateButtons();
}

// ---------------------------------------------------------------- source
async function openPath(p) {
  if (!p) return;
  const isDisc = /\.iso$/i.test(p) || (await window.api.isDir(p));
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
  try {
    status(`Opening ${basename(path)}...`);
    const r = await rpc("probe", { path });
    S.info = r.info; S.detected = r.episodes.map((e) => e.slice()); S.episodes = r.episodes.map((e) => e.slice());
    S.checked = new Set(S.episodes.map((_, k) => k)); S.current = 0; S.queue = {};
    S.order = r.info.order; S.pal_mode = "auto";
    S.out_dir = dirname(path) + "\\restored";
    S.analyzed = false; S.auto = {}; S.frame = null; S.total = 0;
    pictureLoad(pictureFresh());                 // picture settings start from the defaults for every new source
    S.epSet = S.episodes.map(() => null);
    resetCleanup();
    // colour correction belongs to a series: keep it for the next file of the same folder, reset otherwise
    S.layers = []; S.layerSel = -1; S.tracks = 3; S.trackOff = {}; S.src = null; S.vfrAuto = null;   // layers belong to this file
    S.gradeAll = structuredClone(GRADE0);
    S.srcDir = dirname(path); S.levels = null; S.aspect = "auto";
    $("discBox").classList.add("hidden");
    $("dropZone").classList.add("hidden");
    $("srcCard").classList.remove("hidden"); $("epBox").classList.remove("hidden");
    $("anResults").classList.add("hidden");
    showSourceInfo(); renderEpisodes(); sync();
    log(`Opened ${path}: ${S.episodes.length} episode(s)`, "ok");
    status(`Opened ${basename(path)} - ${S.episodes.length} episode${S.episodes.length > 1 ? "s" : ""}. Next: Repair.`, "ok");
    $("placeholder").classList.add("hidden");
    schedulePreview();
  } catch { /* logged */ }
}

function showSourceInfo() {
  const i = S.info, box = $("srcBadges");
  $("srcName").textContent = basename(i.path);
  $("srcName").title = i.path;
  box.innerHTML = "";
  const add = (text, cls) => box.appendChild(el("span", "badge " + (cls || ""), text));
  add(i.pal ? "PAL 25 fps" : "NTSC 29.97 fps", i.pal ? "pal" : "ntsc");
  add(`${i.width}x${i.height}`); add(i.dar > 1.5 ? "16:9" : "4:3"); add(tc(i.frames)); add(i.order === "bff" ? "bottom field first" : "top field first");
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
    const cells = [cb, el("span", null, String(k + 1)), mk(a, 0), mkc(0), mk(b, 1), mkc(1), el("span", null, tc(b - a).slice(2, -4))];
    cells.forEach((c) => { const td = el("td"); td.appendChild(c); tr.appendChild(td); });
    tr.onclick = () => selectEpisode(k);
    tb.appendChild(tr);
  });
  epCount(); sync(); trimInfo(); renderChapters();
}

// ---------------------------------------------------------------- chapters
const chapFrames = () => (S.info && S.info.chapters ? S.info.chapters.map((t) => Math.round(t * fps())) : []);
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
    for (const c of [el("span", "mono", String(i + 1)), t, el("span", "mono", hms((nx - f) / fps()).replace(/^0:0?/, ""))]) {
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
  S.current = k; S.frame = null; S.total = 0; S.src = null; S.vfrAuto = null;
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
                     "stab", "stab_level", "tfix_strength", "crop", "aspect", "aspect_custom", "size", "custom_w", "custom_h", "custom_lock",
                     "kernel", "kernel_b", "kernel_c", "sharp", "upscale", "grade", "layers"]) out[key] = st[key];
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
  frames: (r) => (r.pal_mode ? { pal_mode: r.pal_mode, order: r.order } : { chroma_unmix: r.chroma_unmix, mode: r.mode, comb: r.comb, ivtc_repair: r.ivtc_repair, aa: r.aa }),
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
      : `Telecine & interlacing: ${r.mode === "vfr" ? `VFR - ${r.video_sections} video-rate section(s) found (captions / titles at 30p)` : "CFR 23.976 - film only, no video-rate sections"} `
        + `(${pc(r.video_share)} % of moving cycles). Unmatched frames ${pc(r.unmatched_share)} % (combing ${pc(r.comb_strength)} % of the picture) -> combing repair ${r.comb}, anti-aliasing ${r.aa}`
        + `${r.line_doubled > 40 ? ` (source looks line-doubled: ${r.line_doubled} %)` : ""}. ICP ${r.chroma_unmix === "on" ? "found" : "not found"}.`,
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
    $("autoResult").textContent = msg; log(msg, "ok"); status(msg, "ok");
    sync(); schedulePreview(); scheduleScript();
  } catch { /* logged */ }
  btn.classList.remove("running"); btn.textContent = old;
}

// Colour step: measure black / white level and clipped colour, set them (own button, not part of Analyze)
async function autoColour(wbOnly = false) {
  try {
    status(wbOnly ? "Measuring the colour of the whites..." : "Measuring black level, white level, white balance and clipped colour...");
    const r = await rpc("detect", { kind: "levels", state: backendState([S.current]) });
    if (!r.grade) { status("Nothing to measure (no picture found).", "err"); return; }
    S.levels = r;
    for (const k of (wbOnly ? ["temp", "tint"] : ["black", "white", "gamut", "temp", "tint"])) { S.auto[`grade.${k}`] = r.grade[k]; S.gradeAll[k] = r.grade[k]; }
    sync(); schedulePreview(); scheduleScript();
    const wb = r.grade.temp || r.grade.tint ? `white balance temperature ${r.grade.temp}, tint ${r.grade.tint}` : "white balance neutral";
    const same = r.grade.black === 16 && r.grade.white === 235 && r.grade.gamut === "off";
    const msg = wbOnly ? `Whites: ${r.wb_u == null ? "too few to judge" : `Cb ${sign(r.wb_u)} / Cr ${sign(r.wb_v)} levels`} - ${wb}.`
      : `Colour: darkest tone ${r.level_black}, brightest ${r.level_white}, ${r.gamut_out} % beyond the screen range` +
        (same ? "" : ` - set black ${r.grade.black}, white ${r.grade.white}, clipped colour ${r.grade.gamut}`) + `; ${wb}.`;
    log(msg, "ok"); status(msg, "ok");
  } catch { /* logged */ }
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
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(renderPreview, 200);
  vsviewFollow();
}
// the open vsview follows every change: its job file is rewritten and it reloads itself (same tab and frame)
let vsviewTimer = null;
function vsviewFollow() {
  if (!S.vsviewLive || S.current == null) return;
  clearTimeout(vsviewTimer);
  vsviewTimer = setTimeout(() => rpc("vsview_update", { state: backendState([S.current]) }).catch(() => {}), 700);
}
function previewPos() {
  if (S.total > 1 && S.frame != null) return S.frame / (S.total - 1);
  const r = $("rngPos");
  return r.value / r.max;
}
// ---------------------------------------------------------------- play + trim
const PLAY = { on: false, t: 0, n: 0, last: 0 };
function vfrButton() {
  const b = $("btnVfrPrev"), vfr = S.mode === "vfr" && S.info && !S.info.pal;
  b.classList.toggle("hidden", !S.info);
  const rate = S.rate ? " · " + (Math.abs(S.rate - 23.976) < 0.1 ? "24p" : Math.abs(S.rate - 29.97) < 0.1 ? "30p" : Math.abs(S.rate - 25) < 0.1 ? "25p" : S.rate.toFixed(2)) : "";
  const a = S.vfrAuto || {};
  let t, failed = false;
  if (!S.info) t = "";
  else if (!vfr) t = (S.info.pal ? "PAL preview" : "CFR preview") + rate;
  else if (S.vfrPrev) t = "VFR preview ✓" + rate;
  else if (a.state === "failed") { t = "VFR analysis failed · retry"; failed = true; }
  else t = `VFR analysis ${a.progress ? Math.round(100 * a.progress) + " %" : "…"} · 24p meanwhile`;
  b.textContent = t;
  b.title = !vfr ? "The preview shows the export's frames (constant frame rate)."
    : "VFR: the analysis of this episode runs by itself in the background; until it is done the preview shows 24p, then the export's own frames and frame rates (30 fps video parts such as scrolling captions play smoothly). The export reuses it.";
  b.classList.toggle("primary", failed);
  b.disabled = !failed || S.busy;
}
function autoVfr(r) {
  if (!S.info || S.info.pal || S.mode !== "vfr") { S.vfrAuto = null; vfrButton(); return; }
  if (r.vfr) { S.vfrAuto = { state: "ready" }; vfrButton(); return; }
  rpc("vfr_auto", { state: backendState([S.current]), episode: S.current })
    .then((a) => { if (!(S.vfrAuto && S.vfrAuto.state === "running" && a.state === "running" && !a.progress)) S.vfrAuto = a; vfrButton(); }).catch(() => {});
}
async function onVfrAuto(m) {
  if (m.episode !== undefined && m.episode !== S.current) return;     // an episode no longer shown
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
  if (!PLAY.on) { $("playFps").textContent = ""; window.api.rpc("stop_play", {}); return; }
  if (!S.info) { togglePlay(false); return; }
  if (S.total && S.frame >= S.total - 1) S.frame = 0;
  const views = S.view === "split" ? ["before", "after"] : [S.view];
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
  schedulePreview();
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
const TL = { drag: null, snapPx: 7 };
function tlFrameAtX(x) { const r = $("tl").getBoundingClientRect(); return Math.round(Math.max(0, Math.min(1, (x - r.left) / r.width)) * S.info.frames); }
function tlPct(f) { return (100 * f / Math.max(1, S.info.frames)).toFixed(4) + "%"; }
function headSrcFrame() {                       // source frame shown in the preview
  const [a, b] = S.episodes[S.current] || [0, 0];
  if (!S.total || S.total < 2 || S.frame == null) return a + ($("rngPos").value / $("rngPos").max) * (b - a);
  return a + (S.frame / (S.total - 1)) * (b - a);
}
function renderTimeline() {
  const tl = $("tl");
  if (!S.info || !S.episodes.length) { tl.classList.add("hidden"); return; }
  tl.classList.remove("hidden");
  const eps = $("tlEps"); eps.innerHTML = "";
  S.episodes.forEach(([a, b], k) => {
    if (k === S.current) return;
    const d = el("div"); d.style.left = tlPct(a); d.style.width = tlPct(b - a); d.title = `Episode ${k + 1}: ${tc(a)} – ${tc(b)} (click to open)`;
    d.dataset.ep = k; eps.appendChild(d);
  });
  const ticks = $("tlTicks"); ticks.innerHTML = "";
  for (const f of chapFrames()) { const t = el("div"); t.style.left = tlPct(f); ticks.appendChild(t); }
  const [a, b] = S.episodes[S.current];
  const sel = $("tlSel"); sel.style.left = tlPct(a); sel.style.width = tlPct(b - a);
  $("tlHead").style.left = tlPct(headSrcFrame());
  renderLayerTrack();
}
function tlSnap(f, shift) {
  if (shift) return f;
  const px = $("tl").getBoundingClientRect().width, lim = TL.snapPx / Math.max(1, px) * S.info.frames;
  let best = f, d0 = lim;
  for (const c of [...chapFrames(), 0, S.info.frames]) if (Math.abs(c - f) < d0) { d0 = Math.abs(c - f); best = c; }
  return best;
}
function tlTip(f, x) {
  const tip = $("tlTip"), r = $("tl").getBoundingClientRect();
  tip.textContent = `${tc(f)} · frame ${f}`; tip.style.left = (x - r.left) + "px"; tip.classList.remove("hidden");
}
function tlSeek(f) {
  const [a, b] = S.episodes[S.current];
  const fr = (f - a) / Math.max(1, b - a);
  if (S.total > 1) S.frame = Math.round(Math.max(0, Math.min(1, fr)) * (S.total - 1));
  else { S.frame = null; $("rngPos").max = 1000; $("rngPos").value = Math.round(1000 * Math.max(0, Math.min(1, fr))); }
  renderTimeline(); schedulePreview();
}
function initTimeline() {
  const tl = $("tl");
  tl.addEventListener("pointerdown", (e) => {
    if (!S.info || !S.episodes.length || S.busy) return;
    togglePlay(false);
    const f = tlFrameAtX(e.clientX);
    if (e.target.id === "tlIn" || e.target.id === "tlOut") {
      TL.drag = { kind: e.target.id === "tlIn" ? 0 : 1, el: e.target }; e.target.classList.add("drag");
    } else if (e.target.dataset && e.target.dataset.ep !== undefined) {
      selectEpisode(parseInt(e.target.dataset.ep, 10)); setTimeout(() => tlSeek(f), 0); return;
    } else {
      const [a, b] = S.episodes[S.current];
      if (f < a || f > b) {                      // outside the selected episode: open the one there
        const k = S.episodes.findIndex(([x, y]) => f >= x && f < y);
        if (k >= 0 && k !== S.current) { selectEpisode(k); setTimeout(() => tlSeek(f), 0); }
        return;
      }
      TL.drag = { kind: "seek" }; tlSeek(f);
    }
    try { tl.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    tlTip(f, e.clientX);
  });
  tl.addEventListener("pointermove", (e) => {
    if (!S.info) return;
    const f = tlFrameAtX(e.clientX);
    if (!TL.drag) { tlTip(f, e.clientX); return; }
    const ep = S.episodes[S.current];
    if (TL.drag.kind === "seek") { tlSeek(Math.max(ep[0], Math.min(ep[1], f))); tlTip(f, e.clientX); return; }
    let g = tlSnap(f, e.shiftKey);
    if (TL.drag.kind === 0) g = Math.min(g, ep[1] - 24); else g = Math.max(g, ep[0] + 24);
    ep[TL.drag.kind] = Math.max(0, Math.min(S.info.frames, g));
    renderTimeline(); tlTip(ep[TL.drag.kind], e.clientX);
    trimFields(ep[0], ep[1], true);
  });
  const end = (e) => {
    if (!TL.drag) return;
    const d = TL.drag; TL.drag = null;
    setTimeout(() => $("tlTip").classList.add("hidden"), 600);
    if (d.el) d.el.classList.remove("drag");
    if (d.kind === 0 || d.kind === 1) {          // trim applied: preview the new start / end
      S.frame = d.kind === 0 ? 0 : null; S.total = 0;
      if (d.kind === 1) { $("rngPos").max = 1000; $("rngPos").value = 1000; }
      renderEpisodes(); schedulePreview(); scheduleScript();
      log(`Episode ${S.current + 1}: ${tc(S.episodes[S.current][0])} – ${tc(S.episodes[S.current][1])}`, "ok");
    }
  };
  tl.addEventListener("pointerup", end);
  tl.addEventListener("pointercancel", end);
  tl.addEventListener("pointerleave", () => { if (!TL.drag) $("tlTip").classList.add("hidden"); });
  window.addEventListener("resize", renderTimeline);
}

function trimFields(a, b, live = false) {        // In / Out as times (frame in the tooltip), Length
  for (const [id, v] of [["trimStart", a], ["trimEnd", b]]) {
    const x = $(id);
    if (document.activeElement !== x) x.value = tc(v);
    x.title = x.title.split(" · ")[0] + ` · source frame ${v}`;
    x.classList.toggle("live", live);
  }
  $("tpLen").textContent = tc(b - a).replace(/\.\d+$/, "") + (live ? " · release" : "");
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
  const views = S.view === "split" ? ["before", "after"] : [S.view];
  $("spinner").classList.remove("hidden");
  try {
    const r = await rpc("preview", { state: backendState([S.current]), episode: S.current, pos: previewPos(), views,
                                     frame: S.total && S.frame != null ? S.frame : null });
    if (seq !== previewSeq) return;
    if (r.before) $("imgBefore").src = "data:image/png;base64," + r.before;
    if (r.after) $("imgAfter").src = "data:image/png;base64," + r.after;
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
  } catch { /* superseded or logged */ }
  if (seq === previewSeq) $("spinner").classList.add("hidden");
}
function stepFrame(d) {
  if (!S.total) return;
  S.frame = Math.max(0, Math.min(S.total - 1, (S.frame || 0) + d));
  $("rngPos").value = S.frame;
  showFrameInfo(S.frame, S.total);
  schedulePreview();
}
function setView(v) { const was = PLAY.on; togglePlay(false); S.view = v; setSegView(); if (S.img) schedulePreview(); if (was) setTimeout(() => togglePlay(true), 300); }
function setSegView() { for (const b of $("segView").children) b.classList.toggle("on", b.dataset.v === S.view); }

function layoutImage() {
  if (!S.img) return;
  const stage = $("stage"), wrap = $("imgWrap"), before = $("imgBefore"), after = $("imgAfter");
  const { w, h } = S.img;
  const scale = S.zoom === "Fit" ? Math.min((stage.clientWidth - 4) / w, (stage.clientHeight - 4) / h)
                                 : S.zoom === "1:1" ? 1 : parseInt(S.zoom, 10);
  const W = Math.round(w * scale), H = Math.round(h * scale);
  wrap.style.width = W + "px"; wrap.style.height = H + "px";
  wrap.style.margin = `${Math.max(0, (stage.clientHeight - H) / 2)}px ${Math.max(0, (stage.clientWidth - W) / 2)}px`;
  stage.classList.toggle("pixel", scale >= 1.5 || S.zoom !== "Fit");
  stage.classList.toggle("zoomed", S.zoom !== "Fit");
  for (const im of [before, after]) { im.style.width = W + "px"; im.style.height = H + "px"; }
  const split = S.view === "split";
  before.classList.toggle("hidden", S.view === "after");
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
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); stepFrame((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 24 : 1)); }
    else if (e.key === "b" || e.key === "B") setView("before");
    else if (e.key === "a" || e.key === "A") setView("after");
    else if (e.key === "s" || e.key === "S") setView("split");
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "i" || e.key === "I") trim("in");
    else if (e.key === "o" || e.key === "O") trim("out");
  });
}

// ---------------------------------------------------------------- export
function renderQueue() {
  const q = $("queue");
  q.innerHTML = "";
  const sel = [...S.checked].sort((a, b) => a - b);
  if (!sel.length) { q.appendChild(el("div", "empty", S.info ? "No episodes ticked - tick them in Source." : "Open a source first.")); return; }
  for (const k of sel) {
    const [a, b] = S.episodes[k] || [0, 0];
    const st = S.queue[k] || "waiting";
    const row = el("div", "qrow " + st);
    row.append(el("span", "mono", `#${k + 1}`), el("span", "file", `${tc(a).slice(0, -4)} – ${tc(b).slice(0, -4)}`),
               el("span", "st", { waiting: "waiting", running: "encoding…", done: "done ✓", failed: "failed" }[st]));
    q.appendChild(row);
  }
}
async function exportSelected() {
  const sel = [...S.checked].sort((a, b) => a - b);
  if (!sel.length) return;
  saveSettings();
  S.queue = {}; S.exporting = sel; renderQueue();
  try {
    const st = backendState(sel);
    st.per_episode = {};
    for (const k of sel) st.per_episode[k] = pictureState(k);
    const outs = await rpc("process", { state: st });
    for (const k of sel) S.queue[k] = "done";
    status(`Exported ${outs.length} file${outs.length > 1 ? "s" : ""}`, "ok");
  } catch {
    for (const k of sel) if (S.queue[k] === "running") S.queue[k] = "failed";
  }
  S.exporting = null; renderQueue();
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
  } else if (m.event === "busy") {
    S.busy = m.busy; updateButtons();
    if (!m.busy) $("progBar").style.width = "0";
  } else if (m.event === "stderr") {
    if (!/Warning|DeprecationWarning/.test(m.msg)) log(m.msg, "dim");
  } else if (m.event === "exit") {
    log("Backend stopped - restart the app.", "err"); status("Backend stopped - restart the app.", "err");
  }
}

// ---------------------------------------------------------------- wiring
function bind() {
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
    $("btnAdvanced").textContent = box.classList.contains("hidden") ? "Fine-tune IVTC ▾" : "Fine-tune IVTC ▴";
  };
  $("btnOutDir").onclick = async () => { const p = await window.api.openFolder("Output folder"); if (p) changed("out_dir", p); };
  $("btnShowOut").onclick = () => S.out_dir && window.api.showPath(S.out_dir);
  $("btnExport").onclick = exportSelected;
  $("btnGradeReset").onclick = () => {
    const ly = S.layers[S.layerSel];
    if (ly) { ly.grade = structuredClone(GRADE0); sync(); schedulePreview(); scheduleScript(); return; }   // layer: no changes
    S.grade = structuredClone(GRADE0);
    for (const k of Object.keys(S.auto)) if (k.startsWith("grade.")) delete S.auto[k];
    sync(); schedulePreview(); scheduleScript();
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
  box.addEventListener("input", () => { S.script = box.value; scriptStateUi(); schedulePreview(); });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Tab") { e.preventDefault(); const a = box.selectionStart; box.setRangeText("    ", a, box.selectionEnd, "end"); box.dispatchEvent(new Event("input")); }
  });
  $("btnScriptReset").onclick = () => { S.script = null; refreshScript(true); schedulePreview(); sync(); };
  $("btnScriptCheck").onclick = checkScript;
  $("btnScriptSave").onclick = saveScript;
  $("btnAutoCrop").onclick = autoCrop;
  $("btnReport").onclick = collectReport;
  $("btnVsview").onclick = () => { S.vsviewLive = true; rpc("vsview", { state: backendState([S.current]) }).catch(() => {}); };
  $("btnStop").onclick = () => rpc("cancel", {}).catch(() => {});
  $("btnLog").onclick = () => { const l = $("log"); l.classList.toggle("hidden"); $("btnLog").textContent = l.classList.contains("hidden") ? "Log ▴" : "Log ▾"; layoutImage(); };

  for (const b of $("segView").children) b.onclick = () => setView(b.dataset.v);
  $("btnPlay").onclick = () => togglePlay();
  $("btnVfrPrev").onclick = async () => {
    try { await rpc("vfr_pass", { state: backendState([S.current]), episode: S.current }); S.frame = null; S.total = 0; schedulePreview(); } catch { /* logged */ }
  };
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
  if (S.step !== "script" || S.script) return;           // only follows the settings while not edited
  clearTimeout(scriptTimer); scriptTimer = setTimeout(() => refreshScript(), 300);
}
async function refreshScript(force = false) {
  if (!S.info || (S.script && !force)) { scriptStateUi(); return; }
  try {
    const t = await rpc("script", { state: backendState([S.current]), episode: S.current });
    S.scriptGen = t;
    if (!S.script) $("scriptBox").value = t;
  } catch { /* logged */ }
  scriptStateUi();
}
function scriptStateUi() {
  const ed = !!S.script;
  const chip = $("scriptState");
  chip.className = "chip " + (ed ? "edited" : "auto");
  chip.textContent = ed ? "edited - preview and export use this text" : "from settings";
  $("scriptBox").classList.toggle("edited", ed);
  $("btnScriptReset").disabled = !ed;
  $("scriptFlag").classList.toggle("hidden", !ed);
  $("exportNote").textContent = ed ? "Export uses your edited script (Script step) for every ticked episode." : "";
}
async function checkScript() {
  const msg = $("scriptMsg");
  msg.className = "note"; msg.textContent = "Running the script...";
  try {
    const r = await rpc("check_script", { state: backendState([S.current]), episode: S.current, text: $("scriptBox").value });
    if (r.ok) { msg.className = "note ok"; msg.textContent = `OK: ${r.frames} frames, ${r.width}x${r.height}, ${r.format}, ${r.fps} fps (preview rate).`; }
    else { msg.className = "note err"; msg.textContent = `${r.error}${r.where ? "  -  " + r.where : ""}`; }
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
    sl.ondblclick = () => changed(key, def);
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
      let u = (e.clientX - r.left - size / 2) / R, v = -(e.clientY - r.top - size / 2) / R;
      if (fine && start) { u = start[1] + (u - start[3]) * 0.2; v = start[2] + (v - start[4]) * 0.2; }
      const m = Math.hypot(u, v); if (m > 1) { u /= m; v /= m; }
      return [wheelVal(name)[0], +u.toFixed(3), +v.toFixed(3)];
    };
    cv.onpointerdown = (e) => {
      drag = true; fine = e.shiftKey; cv.setPointerCapture(e.pointerId);
      if (fine) { const r = cv.getBoundingClientRect(), R = size / 2 - 6; const cur = wheelVal(name); start = [0, cur[1], cur[2], (e.clientX - r.left - size / 2) / R, -(e.clientY - r.top - size / 2) / R]; }
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
  if (S.step !== "colour" || !S.info) return;
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
  if (add.length) changed("grade.luts", [...lutList(), ...add]);
}
function lutList() {
  const g = S.grade;
  if (g.lut) { g.luts = [{ path: g.lut, mix: g.lut_mix ?? 1, on: true }, ...(g.luts || [])]; g.lut = ""; }   // older settings
  return (g.luts || []).map((l) => ({ ...l }));
}
let LUT_SIG = "";
function lutUi() {
  const L = lutList(), sig = JSON.stringify(L);
  if (sig === LUT_SIG && $("lutList").childElementCount === L.length) return;
  LUT_SIG = sig;
  const box = $("lutList"); box.innerHTML = "";
  const set = (arr) => changed("grade.luts", arr);
  L.forEach((l, i) => {
    const row = el("div", "mrow" + (l.on === false ? " off" : ""));
    row.append(el("div", "num", String(i + 1)));
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
function epLayers() { const [a, b] = epRange(); return S.layers.map((l, i) => ({ l, i })).filter(({ l }) => l.start < b && l.end > a); }
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
  const f = Math.max(a, Math.min(b - 2, at ?? playhead())), len = Math.round(5 * fps());
  let t = track, g = null;
  if (t) g = gapAt(t, f);
  else for (t = 1; t <= S.tracks; t++) { g = gapAt(t, f); if (g && g[1] - f >= 2) break; }
  if (!g || g[1] - f < 2) { if (track) return; t = ++S.tracks; g = [a, b]; }
  const n = S.layers.length + 1;
  S.layers.push({ name: `Adjustment Layer ${n}`, track: t, start: f, end: Math.min(g[1], f + len), on: true, opacity: 1,
                  grade: structuredClone(GRADE0), col: LAYER_COL[(n - 1) % LAYER_COL.length] });
  S.layerSel = S.layers.length - 1;
  layersChanged();
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
  if (!l || S.src == null || (S.src >= l.start && S.src < l.end)) return;
  try {                                              // the frame shown is outside the layer: go to it
    const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: Math.max(l.start, epRange()[0]) + 2 });
    togglePlay(false); S.frame = r.frame; schedulePreview();
  } catch { /* logged */ }
}
function moveTrack(i, d) {
  const l = S.layers[i], t = (l.track || 1) + d;
  if (t < 1) return;
  if (t > S.tracks) S.tracks = t;
  if (!trackFree(t, l.start, l.end, l)) { status(`Track L${t} is not free there.`); return; }
  l.track = t; layersChanged();
}
function removeLayer(i, ask = true) {
  const l = S.layers[i];
  if (ask && !sameGrade(l.grade, GRADE0) && !confirm(`Delete "${l.name}" and its grade?`)) return;
  S.layers.splice(i, 1);
  if (S.layerSel === i) S.layerSel = -1; else if (S.layerSel > i) S.layerSel--;
  layersChanged();
}
let LAYER_SIG = "";
function layerUi() {
  const sel = S.layers[S.layerSel], list = epLayers();
  if (sel) {
    const out = S.src != null && (S.src < sel.start || S.src >= sel.end);
    $("layerTarget").innerHTML = `Editing: <b>${sel.name.replace(/[<&]/g, "")}</b> · track L${sel.track || 1} · ${tc(sel.start)} – ${tc(sel.end)}` +
      (sel.on === false || S.trackOff[sel.track || 1] ? ' · <span class="warn">switched off</span>' : "") + (out ? ' · <span class="warn">the frame shown is outside this layer</span>' : "");
  } else {
    $("layerTarget").innerHTML = "Editing: <b>Base grade</b> (whole video)" + (list.length ? ` · ${list.length} adjustment layer${list.length === 1 ? "" : "s"} on top` : "");
  }
  $("btnGradeReset").textContent = sel ? "Reset layer" : "Reset all";
  $("btnLayerAdd").disabled = !S.info;
  const sig = JSON.stringify([list.map(({ l, i }) => [i, l.name, l.track, l.start, l.end, l.on, l.opacity, l.col]), S.layerSel, S.current, S.layers.length, S.trackOff]);
  if (sig === LAYER_SIG) return;
  LAYER_SIG = sig;
  const box = $("layerList"); box.innerHTML = "";
  const sorted = [...list].sort((p, q) => (q.l.track || 1) - (p.l.track || 1) || p.l.start - q.l.start);   // top track first
  for (const { l, i } of sorted) {
    const row = el("div", "layer-row" + (i === S.layerSel ? " sel" : "") + (l.on === false || S.trackOff[l.track || 1] ? " off" : ""));
    const l1 = el("div", "l1"), sw = el("span", "sw"); sw.style.background = l.col || LAYER_COL[0];
    const nm = el("input", "lname"); nm.type = "text"; nm.value = l.name; nm.title = "Name (click to rename)";
    nm.onchange = () => { l.name = nm.value.trim() || l.name; layersChanged(); };
    nm.onclick = (e) => { e.stopPropagation(); if (S.layerSel !== i) selectLayer(i); };
    const btns = el("div", "lbtns");
    const b = (txt, title, fn, dis) => { const x = el("button", null, txt); x.title = title; x.disabled = !!dis; x.onclick = (e) => { e.stopPropagation(); fn(); }; btns.appendChild(x); };
    b(l.on === false ? "Off" : "On", "Switch this layer on / off", () => { l.on = l.on === false; layersChanged(); });
    b("↑", "Move to the track above (works on top of more)", () => moveTrack(i, 1));
    b("↓", "Move to the track below", () => moveTrack(i, -1), (l.track || 1) <= 1);
    b("✕", "Delete this layer", () => removeLayer(i));
    l1.append(sw, nm, btns);
    const l2 = el("div", "l2");
    const r = el("input"); r.type = "range"; r.min = 0; r.max = 1; r.step = 0.01; r.value = l.opacity ?? 1;
    const t = el("input"); t.type = "text"; t.value = Math.round(100 * (l.opacity ?? 1)) + "%";
    const put = (v) => { l.opacity = Math.max(0, Math.min(1, +(+v).toFixed(2))); layersChanged(); };
    r.oninput = () => { t.value = Math.round(100 * r.value) + "%"; };
    r.onchange = () => put(r.value);
    t.onchange = () => { const v = parseFloat(t.value); put(isNaN(v) ? l.opacity ?? 1 : v / 100); };
    r.ondblclick = () => put(1);
    for (const x of [r, t]) x.onclick = (e) => e.stopPropagation();
    const at = (txt, title, fn, dis) => { const x = el("button", null, txt); x.title = title; x.disabled = !!dis; x.onclick = (e) => { e.stopPropagation(); fn(); }; return x; };
    const here = S.src;
    l2.append(el("span", null, `L${l.track || 1} · ${tc(l.start)} – ${tc(l.end)} · ${l.end - l.start} fr · opacity`), r, t,
      at("Start here", "Layer starts at the frame shown", () => { if (trackFree(l.track || 1, here, l.end, l)) { l.start = here; layersChanged(); } }, here == null || here >= l.end),
      at("End here", "Layer ends after the frame shown", () => { if (trackFree(l.track || 1, l.start, here + 1, l)) { l.end = here + 1; layersChanged(); } }, here == null || here < l.start));
    row.append(l1, l2);
    row.onclick = () => selectLayer(i);
    box.appendChild(row);
  }
  const base = el("div", "layer-row base" + (S.layerSel < 0 ? " sel" : ""));
  const l1 = el("div", "l1"), sw = el("span", "sw"); sw.style.background = "#3b5378";
  const nm = el("input", "lname"); nm.value = "Base grade · whole video (V1)"; nm.readOnly = true; nm.tabIndex = -1;
  l1.append(sw, nm); base.append(l1);
  base.onclick = () => selectLayer(-1);
  box.appendChild(base);
  const other = S.layers.length - list.length;
  if (other) box.appendChild(el("div", "scene-empty", `${other} layer${other === 1 ? "" : "s"} in other episodes`));
}

// ---- the multi-track timeline under the picture (Colour step)
const LT = { zoom: 60, v0: null, drag: null, seekF: null, manual: false };
function ltWindow() {
  const [a, b] = epRange();
  if (!LT.zoom) return [a, b];
  const len = Math.min(b - a, Math.round(LT.zoom * fps()));
  const h = LT.seekF ?? S.src ?? Math.round(headSrcFrame());
  if (LT.v0 == null || (!LT.drag && !LT.manual && (h < LT.v0 || h > LT.v0 + len))) LT.v0 = h - len * 0.25;
  LT.v0 = Math.max(a, Math.min(b - len, LT.v0));
  return [LT.v0, LT.v0 + len];
}
function ltX(f, w) { return (100 * (f - w[0]) / Math.max(1, w[1] - w[0])); }
function ltFrame(x) { const r = $("ltBody").getBoundingClientRect(), w = ltWindow(); return Math.round(w[0] + Math.max(0, Math.min(1, (x - r.left) / r.width)) * (w[1] - w[0])); }
function ltTrackAt(y) {
  for (const lane of $$("#ltLanes .lt-lane[data-track]")) { const r = lane.getBoundingClientRect(); if (y >= r.top && y < r.bottom) return +lane.dataset.track; }
  return null;
}
function renderLayerTrack() {
  const lt = $("lt");
  if (!lt) return;
  const show = S.step === "colour" && !!S.info && S.episodes.length > 0;
  lt.classList.toggle("hidden", !show);
  if (!show) return;
  const maxT = Math.max(...S.layers.map((l) => l.track || 1), 1);
  if (S.tracks < maxT) S.tracks = maxT;
  for (const z of $("ltZoom").children) z.classList.toggle("on", +z.dataset.v === LT.zoom);
  const w = ltWindow(), span = w[1] - w[0], [a, b] = epRange();
  const ruler = $("ltRuler"); ruler.innerHTML = "";
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900].map((s) => s * fps()).find((s) => span / s <= 10) || 1800 * fps();
  for (let f = Math.ceil(w[0] / step) * step; f < w[1]; f += step) {
    const lab = el("span", "lab", hms(f / fps()).replace(/^0:/, "")); lab.style.left = ltX(f, w) + "%"; ruler.appendChild(lab);
    const d = el("div", "cut"); d.style.left = ltX(f, w) + "%"; ruler.appendChild(d);
  }
  const heads = $("ltHeads"); heads.innerHTML = ""; heads.appendChild(el("div", "sp"));
  const lanes = $("ltLanes"); lanes.innerHTML = "";
  for (let t = S.tracks; t >= 1; t--) {
    const off = !!S.trackOff[t];
    const th = el("div", "th" + (off ? " off" : ""), `L${t}`);
    const eye = el("button", null, off ? "◌" : "◉"); eye.title = off ? "Show this track (its layers are used again)" : "Hide this track (its layers are not used)";
    eye.onclick = () => { if (off) delete S.trackOff[t]; else S.trackOff[t] = true; layersChanged(); };
    th.appendChild(eye); heads.appendChild(th);
    const lane = el("div", "lt-lane" + (off ? " off" : "")); lane.dataset.track = t;
    for (const { l, i } of epLayers()) {
      if ((l.track || 1) !== t) continue;
      const x0 = Math.max(-1, ltX(l.start, w)), x1 = Math.min(101, ltX(l.end, w));
      if (x1 <= 0 || x0 >= 100) continue;
      const bar = el("div", "lt-layer" + (i === S.layerSel ? " sel" : "") + (l.on === false ? " off" : ""), l.name);
      bar.style.left = x0 + "%"; bar.style.width = Math.max(0.3, x1 - x0) + "%"; bar.style.background = l.col || LAYER_COL[0];
      bar.dataset.i = i; bar.title = `${l.name} · L${t}: ${tc(l.start)} – ${tc(l.end)} · opacity ${Math.round(100 * (l.opacity ?? 1))} %`;
      lane.appendChild(bar);
    }
    lanes.appendChild(lane);
  }
  heads.appendChild(el("div", "th v", "V1"));
  const v = el("div", "lt-lane video"), clip = el("div", "vclip", `${S.info.path ? basename(S.info.path) : "video"} · episode ${S.current + 1} · base grade`);
  clip.style.left = Math.max(0, ltX(a, w)) + "%"; clip.style.width = (Math.min(100, ltX(b, w)) - Math.max(0, ltX(a, w))) + "%";
  v.appendChild(clip); lanes.appendChild(v);
  const h = LT.seekF ?? S.src ?? headSrcFrame();
  $("ltHead").style.left = ltX(h, w) + "%";
  $("ltHead").classList.toggle("hidden", h < w[0] || h > w[1]);
}
function ltSnap(f, shift, skip) {
  if (shift) return f;
  const w = ltWindow(), px = $("ltBody").getBoundingClientRect().width, lim = 7 / Math.max(1, px) * (w[1] - w[0]);
  const cands = [...epRange(), ...(S.src != null ? [S.src, S.src + 1] : [])];
  for (const l of S.layers) if (l !== skip) cands.push(l.start, l.end);
  let best = f, d0 = lim;
  for (const c of cands) if (Math.abs(c - f) < d0) { d0 = Math.abs(c - f); best = c; }
  return best;
}
function ltTip(text, x) { const tip = $("ltTip"), r = $("ltBody").getBoundingClientRect(); tip.textContent = text; tip.style.left = (x - r.left) + "px"; tip.style.top = "2px"; tip.classList.remove("hidden"); }
async function ltSeekExact(f) {
  try {
    const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: f });
    S.frame = r.frame; schedulePreview();
  } catch { /* logged */ }
}
function initLayerTrack() {
  const body = $("ltBody");
  for (const z of $("ltZoom").children) z.onclick = () => { LT.zoom = +z.dataset.v; LT.v0 = null; LT.manual = false; renderLayerTrack(); };
  $("ltAdd").onclick = () => addLayer();
  $("ltSplit").onclick = splitLayer;
  $("ltTrack").onclick = () => { S.tracks++; renderLayerTrack(); };
  body.addEventListener("pointerdown", (e) => {
    if (!S.info || e.button === 2) return;
    togglePlay(false);
    const f = ltFrame(e.clientX), bar = e.target.closest(".lt-layer");
    if (bar) {
      const i = +bar.dataset.i, l = S.layers[i], r = bar.getBoundingClientRect();
      const edge = Math.min(7, r.width / 4);                // short bars: most of the bar moves it
      const kind = e.clientX - r.left < edge ? "l" : r.right - e.clientX < edge ? "r" : "move";
      LT.drag = { kind, i, f0: f, s0: l.start, e0: l.end, t0: l.track || 1, moved: false };
      if (S.layerSel !== i) { S.layerSel = i; sync(); }
    } else {
      LT.drag = { kind: "seek" }; LT.manual = false; LT.seekF = f; tlSeek(f);
    }
    try { body.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    renderLayerTrack();
  });
  body.addEventListener("pointermove", (e) => {
    if (!S.info) return;
    const f = ltFrame(e.clientX);
    if (!LT.drag) { ltTip(`${tc(f)} · frame ${f}`, e.clientX); return; }
    const d = LT.drag, [a, b] = epRange();
    if (d.kind === "seek") { LT.seekF = Math.max(a, Math.min(b - 1, f)); tlSeek(LT.seekF); ltTip(tc(LT.seekF), e.clientX); return; }
    const l = S.layers[d.i], minLen = 2;
    if (d.kind === "move") {
      const len = d.e0 - d.s0;
      let s = d.s0 + (f - d.f0);
      const s2 = ltSnap(s, e.shiftKey, l), e2 = ltSnap(s + len, e.shiftKey, l);
      s = s2 !== s ? s2 : e2 !== s + len ? e2 - len : s;
      s = Math.max(a, Math.min(b - len, s));
      const t = ltTrackAt(e.clientY) ?? l.track ?? 1;
      if (trackFree(t, s, s + len, l)) { l.start = s; l.end = s + len; l.track = t; }
      else if (trackFree(l.track || 1, s, s + len, l)) { l.start = s; l.end = s + len; }     // other track busy: stay on this one
    } else {
      const g = gapAt(l.track || 1, d.kind === "l" ? l.end - 1 : l.start, l) || [a, b];   // neighbours on the track limit trimming
      if (d.kind === "l") l.start = Math.max(g[0], Math.min(l.end - minLen, ltSnap(f, e.shiftKey, l)));
      else l.end = Math.min(g[1], Math.max(l.start + minLen, ltSnap(f, e.shiftKey, l)));
    }
    d.moved = true;
    ltTip(`L${l.track || 1} · ${tc(l.start)} – ${tc(l.end)} · ${l.end - l.start} fr`, e.clientX);
    renderLayerTrack();
  });
  const end = () => {
    const d = LT.drag; LT.drag = null;
    setTimeout(() => $("ltTip").classList.add("hidden"), 600);
    if (!d) return;
    if (d.kind === "seek") { const f = LT.seekF; LT.seekF = null; if (f != null) ltSeekExact(f); return; }
    if (d.moved) layersChanged(); else selectLayer(d.i);
  };
  body.addEventListener("pointerup", end);
  body.addEventListener("pointercancel", end);
  body.addEventListener("pointerleave", () => { if (!LT.drag) $("ltTip").classList.add("hidden"); });
  body.addEventListener("dblclick", (e) => {
    if (e.target.closest(".lt-layer")) return;
    const t = ltTrackAt(e.clientY);
    if (t) addLayer(t, ltFrame(e.clientX));
  });
  body.addEventListener("wheel", (e) => {
    if (!S.info) return;
    e.preventDefault();
    const Z = [0, 300, 60, 15];
    if (e.ctrlKey) {
      const k = Math.max(0, Math.min(Z.length - 1, Z.indexOf(LT.zoom) + (e.deltaY < 0 ? 1 : -1)));
      LT.zoom = Z[k]; LT.v0 = null; LT.manual = false;
    } else if (LT.zoom) {
      const w = ltWindow(); LT.v0 = w[0] + (e.deltaY + e.deltaX) / 600 * (w[1] - w[0]); LT.manual = true;   // stays where scrolled
    }
    renderLayerTrack();
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (S.step !== "colour" || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if ((e.key === "Delete" || e.key === "Backspace") && S.layers[S.layerSel]) { e.preventDefault(); removeLayer(S.layerSel); }
    else if ((e.key === "k" || e.key === "K") && e.ctrlKey) { e.preventDefault(); splitLayer(); }
  });
}

function initColourUi() {
  buildSliders(); buildWheels(); initCurves(); initScopes();
  $("btnLutLoad").onclick = async () => loadLuts(await window.api.openLut());
  $("btnLayerAdd").onclick = () => addLayer();
  initLayerTrack();
}
function syncColourUi() { lutUi(); layerUi(); syncSliders(); for (const w of $$(".wheel[data-wheel]")) if (w._bg) drawWheel(w); drawCurves(); }

async function init() {
  bind();
  let hello;
  for (let i = 0; i < 50; i++) {                       // backend may still be starting
    try { hello = await rpc("hello", {}); break; } catch { await new Promise((r) => setTimeout(r, 200)); }
  }
  if (!hello) { status("Could not start the Python backend.", "err"); return; }
  S.opts = hello.options;
  for (const k of SAVED) if (hello.settings && k in hello.settings) S[k] = hello.settings[k];
  if (!S.opts.ai_model.some((o) => o[0] === S.ai_model)) S.ai_model = "precise";   // UniNet / StableNet were removed from the menu
  buildControls(); buildAdvanced(); initColourUi(); initTempUi(); applyStageOrder(); initStageDrag(); goto("source"); sync();

  if (hello.missing.length) { log("Missing tools: " + hello.missing.join(", "), "err"); status("Missing tools: " + hello.missing.join(", "), "err"); }
  else status("Ready - open or drop a DVD or video.");
}
init();
