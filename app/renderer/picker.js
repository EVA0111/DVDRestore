// The library window (Repair › + Add / + VapourSynth filters): categories on the left, search + quick chips + filter
// cards in the middle, the selected filter on the right with a before / after of the frame shown (server
// m_filter_preview), its settings and Add. Everything addable: the app's stages that can run as an extra step, code
// templates, VapourSynth plugins, vs-jetpack, your AI models and your own filters (VSF_LIB).
// VSFB (app.js) holds the target group and an optional onAdd callback.
"use strict";

const PICK = { items: [], cats: [], cat: "all", chip: "all", sel: null, showUntested: false, prev: {}, prevSeq: 0, prevTimer: 0, split: 50,
               eng: {}, engOpen: {}, building: null };     // eng: model path -> the engine settings chosen in the library (used when the model is added)
// app stages that make sense as an extra step anywhere (the ones every chain has - source, IVTC, output, the grade,
// crop / size - stay in their Guided panels)
const PICK_EVA_OK = new Set(["deblend_ai", "aa", "cleanup", "xtalk", "deblock", "halos", "chroma", "neutral", "stab", "still_lock",
                             "sharpen", "lines", "format", "jet_qtgmc", "jet_dfttest", "jet_aa"]);
const PICK_TEMPLATES = [
  { name: "Mask merge (two ways through a mask)", desc: "Strong denoise in flat areas, the original lines kept - a branch and a mask in a few lines.",
    code: "import vstools, vsmasktools\nstrong = core.zsmooth.FluxSmoothST(clip, temporal_threshold=12, spatial_threshold=12)   # way 1: strong\nmask = vsmasktools.Kirsch.edgemask(vstools.get_y(clip)).std.Maximum()   # the outlines\nclip = core.std.MaskedMerge(strong, clip, mask, first_plane=True)" },
  { name: "Only some frames (a frame range)", desc: "A filter on source frames A to B only (as numbered on the Timeline).",
    code: "alt = core.std.BoxBlur(clip, hradius=2, vradius=2)            # your filter here\nclip = dr.splice_src(clip, alt, [(1000, 1500)])" },
  { name: "Tile an AI model", desc: "Split the picture into tiles for a model with a size or VRAM limit, then put it back together (vs_tiletools).",
    code: "import vs_tiletools\nclip = vs_tiletools.tile(clip, width=256, height=256, overlap=16)\n# clip = ... your model on the tiles ...\nclip = vs_tiletools.untile(clip, fade=True)" },
];
const PICK_ICON = { eva: ["◆", "#5b6cf0"], tpl: ["{ }", "#7a6a2c"], ai: ["✦", "#b4566a"], my: ["★", "#b4566a"], den: ["≈", "#2f8f86"], deb: ["▤", "#7a6a2c"],
  dbk: ["▦", "#6d6a9e"], ana: ["◍", "#a0773a"], dhl: ["◌", "#2f8f86"], shp: ["◆", "#a0773a"], aa: ["⟋", "#3b6b9e"], ivtc: ["≡", "#3b5378"],
  mo: ["↝", "#3b6b9e"], blur: ["◐", "#555e74"], rs: ["⤢", "#555e74"], col: ["◑", "#a05a8a"], geo: ["⛶", "#555e74"], frm: ["⧗", "#555e74"],
  mask: ["◧", "#8a5fbf"], meas: ["▥", "#555e74"], txt: ["T", "#555e74"], adv: ["ƒ", "#555e74"], dist: ["∿", "#b4566a"], tile: ["▦", "#7a6a2c"] };
const PICK_CHIPS = [["all", "All"], ["gpu", "Runs on the GPU"], ["temporal", "Temporal"], ["fast", "Fast"], ["setup", "Needs setup"], ["fav", "★ Favorites"], ["recent", "Recent"]];
function pickStore(k, d) { try { return JSON.parse(localStorage.getItem(k) || "null") || d; } catch { return d; } }
function pickSave(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* no storage */ } }
function pickId(f) { return f.pickId || `${f.kind}:${f.call || ""}:${f.path || f.id || ""}`; }
function pickSrc(f) {
  return { eva: "EVA stage", tpl: "Code template", ai: "AI model", my: "Your filter", node: "Your custom filter", code: "Python", jet: "vs-jetpack" }[f.kind]
    || "Plugin · " + (f.call || "").split(".")[0];
}
function pickDesc(f) { return (f.doc || f.note || "").trim() || (["eva", "tpl"].includes(f.kind) ? "" : vsfCatInfo(f.group)[1]); }
function pickText(f) { return `${f.name} ${f.call || ""} ${pickDesc(f)} ${(f.args || []).map((a) => a.name).join(" ")}`; }
function pickTraits(f) {                              // badges: GPU, temporal, slow, needs setup
  const t = pickText(f).toLowerCase(), c = (f.call || "").toLowerCase();
  const gpu = f.kind === "ai" || /cuda|tensorrt|\btrt\b|nvidia|dlvfx|undistort|bm3dcuda|vsmlrt|\bai\b|net\)/.test(t)
    || (f.kind === "eva" && /ai|net/i.test(`${f.name} ${f.doc}`));
  const temporal = f.group === "mo" || /temporal|degrain|tmedian|ttemp|stabil|still|undistort|overlaps|\btr\b|tr0|thsad|radius.*frames|cleanup/.test(t)
    || (f.args || []).some((a) => /^(tr|tr\d|tradius|temporal_radius|temp_window|delta|frames)$/.test(a.name));
  const slow = /eedi3|bm3d|undistort|qtgmc|nlmeans|knlm|dfttest|fsr|sangnom|\bai\b/.test(t) || f.kind === "ai";
  return { gpu, temporal, slow, setup: f.status === "args" };
}
function pickBuild() {
  if (PICK.only === "ai") {                             // + Add a model: only your AI models (the folders / files they come from are set at the top)
    PICK.items = (VSF_LIB || []).filter((f) => f.kind === "ai").map((f) => ({ ...f, pickId: pickId(f) }));
    const g = (VSF_GROUPS || []).find((x) => x[0] === "ai");
    PICK.cats = [{ key: "ai", name: "Your AI models", desc: g ? g[2] : "", sec: "Models" }];
    return;
  }
  const items = [];
  for (const T of (CHM.lib && CHM.lib.blocks) || []) {
    if (!PICK_EVA_OK.has(T.id)) continue;
    if (S.info && T.ntsc === true && S.info.pal) continue;
    if (S.info && T.ntsc === false && !S.info.pal) continue;
    items.push({ kind: "eva", id: T.id, pickId: "eva:" + T.id, name: T.name, group: "eva", doc: T.desc, status: "ok", block: T });
  }
  for (const t of PICK_TEMPLATES) items.push({ kind: "tpl", pickId: "tpl:" + t.name, name: t.name, group: "tpl", doc: t.desc, code: t.code, status: "ok" });
  PICK.items = [...items, ...(VSF_LIB || []).filter((f) => f.kind !== "ai").map((f) => ({ ...f, pickId: pickId(f) }))];   // AI models: + Add a model
  PICK.cats = [{ key: "eva", name: "EVA stages", desc: "The app's own repair steps, as an extra step anywhere", sec: "EVA toolkit" },
               { key: "tpl", name: "Code templates", desc: "Ready-made code: mask merge, a frame range, tiling", sec: "EVA toolkit" },
               ...(VSF_GROUPS || []).filter(([key]) => key !== "ai").map(([key, name, desc]) => ({ key, name, desc, sec: "Filters" }))];
}
function pickWords() { return $("vsfDlgSearch").value.toLowerCase().split(/\s+/).filter(Boolean); }
function pickMatch(f, words) {
  if (!words.length) return true;
  const c = PICK.cats.find((x) => x.key === f.group);
  const t = `${pickText(f)} ${c ? c.name + " " + c.desc : ""} ${pickSrc(f)}`.toLowerCase();
  return words.every((w) => t.includes(w));
}
function pickUsable(f) { return PICK.showUntested || !f.status || f.status === "ok" || f.status === "args" || f.kind === "my"; }
function pickChipOk(f, fav, recent) {
  const tr = pickTraits(f);
  return PICK.chip === "all" || (PICK.chip === "gpu" && tr.gpu) || (PICK.chip === "temporal" && tr.temporal) || (PICK.chip === "fast" && !tr.slow && !tr.gpu)
    || (PICK.chip === "setup" && tr.setup) || (PICK.chip === "fav" && fav.includes(f.pickId)) || (PICK.chip === "recent" && recent.includes(f.pickId));
}
function pickRender() {
  const words = pickWords(), fav = pickStore("pickFav", []), recent = pickStore("pickRecent", []);
  const pool = PICK.items.filter((f) => pickMatch(f, words) && pickChipOk(f, fav, recent));
  const usable = pool.filter(pickUsable), hiddenN = pool.length - usable.length;
  // categories
  const nav = $("vsfDlgCats"); nav.textContent = "";
  const counts = {}; for (const f of usable) counts[f.group] = (counts[f.group] || 0) + 1;
  const catBtn = (key, name, n, tip, icon) => {
    const b = el("button", "pick-cat" + (PICK.cat === key ? " on" : "") + (n ? "" : " empty"));
    const ic = el("i", null, icon ? icon[0] : ""); if (icon) ic.style.background = icon[1];
    b.append(ic, el("span", null, name), el("small", null, String(n))); b.title = tip || "";
    b.onclick = () => { PICK.cat = key; VSFB.cat = key; pickRender(); $("vsfDlgSearch").focus(); };
    nav.appendChild(b);
  };
  if (!PICK.only) catBtn("all", "All", usable.length, "Everything you can add here", ["∗", "#3a4152"]);
  let sec = null;
  for (const c of PICK.cats) {
    if (c.sec !== sec) { sec = c.sec; nav.appendChild(el("div", "pick-sec", sec)); }
    catBtn(c.key, c.name, counts[c.key] || 0, c.desc, PICK_ICON[c.key] || ["•", "#555e74"]);
  }
  // chips
  const chips = $("pickChips"); chips.textContent = "";
  for (const [k, t] of PICK_CHIPS) {
    const b = el("button", PICK.chip === k ? "on" : "", t);
    b.onclick = () => { PICK.chip = PICK.chip === k && k !== "all" ? "all" : k; pickRender(); };
    chips.appendChild(b);
  }
  // cards
  const list = $("vsfDlgRows"); list.textContent = "";
  if (!VSF_LIB) { list.appendChild(el("div", "pick-empty", "Loading the filter list… (the first time after plugins changed, every filter is tested once: up to a minute)")); pickFoot(0); pickSide(); return; }
  const shown = usable.filter((f) => PICK.cat === "all" || f.group === PICK.cat);
  const order = PICK.cats.map((c) => c.key);
  shown.sort((a, b) => (order.indexOf(a.group) - order.indexOf(b.group)) || a.name.localeCompare(b.name));
  if (PICK.cat === "all" && !words.length && PICK.chip === "all") {
    const code = PICK.items.find((f) => f.kind === "code");
    if (code) { const nb = el("button", "pick-new"); nb.innerHTML = "<b>+</b> Custom / new filter <span>a few lines of your own VapourSynth code</span>"; nb.onclick = () => pickAdd(code); list.appendChild(nb); }
  }
  let last = null, grid = null;
  for (const f of shown) {
    if (f.group !== last) {
      last = f.group;
      const c = PICK.cats.find((x) => x.key === f.group);
      if (PICK.cat === "all") { const h = el("div", "pick-head", c ? c.name : f.group); if (c && c.desc) h.appendChild(el("span", null, " · " + c.desc)); list.appendChild(h); }
      grid = el("div", "pick-grid"); list.appendChild(grid);
    }
    grid.appendChild(pickCard(f, fav));
  }
  if (!shown.length && PICK.only === "ai" && !words.length && PICK.chip === "all")
    list.appendChild(el("div", "pick-empty", "No models yet. Use + Add model folder (the folder that holds your .onnx / .pth / .safetensors files) or + Add model file above."));
  else if (!shown.length) list.appendChild(el("div", "pick-empty", PICK.chip === "fav" ? "No favorites yet: ☆ on a filter stars it." : "Nothing found - try other words, another chip or All."));
  if (hiddenN) {
    const t = el("button", "link pick-more", PICK.showUntested ? "Hide the untested filters" : `Show ${hiddenN} untested filter${hiddenN > 1 ? "s" : ""} (not checked on this computer: may need settings or not work)`);
    t.onclick = () => { PICK.showUntested = !PICK.showUntested; pickRender(); };
    list.appendChild(t);
  }
  vsfAiBar(); vsfMyBar();
  pickFoot(shown.length, fav.length);
  if (PICK.sel && !shown.some((f) => f.pickId === PICK.sel.pickId)) PICK.sel = null;
  pickSide();
  const s = list.querySelector(".pick-card.sel"); if (s) s.scrollIntoView({ block: "nearest" });
}
function pickFoot(n, favN = pickStore("pickFav", []).length) { $("pickCount").textContent = `${n} shown · ${favN} favorite${favN === 1 ? "" : "s"}`; }
function pickCard(f, fav) {
  const sel = PICK.sel && PICK.sel.pickId === f.pickId;
  const card = el("div", "pick-card" + (sel ? " sel" : ""));
  card.dataset.id = f.pickId;
  const [g, col] = PICK_ICON[f.kind === "ai" ? "ai" : f.group] || ["•", "#555e74"];
  const ic = el("div", "pick-ico", g); ic.style.background = col;
  const name = el("b", null, f.name || f.call);
  const star = el("button", "pick-star" + (fav.includes(f.pickId) ? " on" : ""), fav.includes(f.pickId) ? "★" : "☆");
  star.title = "Favorite"; star.onclick = (e) => { e.stopPropagation(); const v = pickStore("pickFav", []); pickSave("pickFav", v.includes(f.pickId) ? v.filter((x) => x !== f.pickId) : [...v, f.pickId]); pickRender(); };
  const top = el("div", "pick-ctop"); top.append(name, star);
  const tr = pickTraits(f), badges = el("div", "pick-badges");
  badges.appendChild(el("span", "bd src", pickSrc(f)));
  if (tr.gpu) badges.appendChild(el("span", "bd gpu", "GPU"));
  if (tr.temporal) badges.appendChild(el("span", "bd tmp", "Temporal"));
  if (tr.slow) badges.appendChild(el("span", "bd slow", "Slow"));
  if (tr.setup) badges.appendChild(el("span", "bd warn", "Needs setup"));
  else if (f.status && f.status !== "ok" && f.kind !== "my") badges.appendChild(el("span", "bd dim", "untested"));
  card.append(ic, top, el("p", null, pickDesc(f) || " "), badges);
  card.onclick = () => { PICK.sel = f; VSFB.sel = f; for (const c of $$("#vsfDlgRows .pick-card")) c.classList.toggle("sel", c === card); pickSide(); };
  card.ondblclick = () => pickAdd(f);
  return card;
}
// ---- the right side: before / after of the frame shown, what it does, its settings, Add
function pickPreviewSpec(f) {
  if (f.kind === "eva") return { type: f.id, p: {} };
  if (f.kind === "tpl" || f.kind === "node") return { type: "code", p: { code: f.code } };
  const p = { call: f.call, args: { ...(f.defaults || {}) } };
  for (const k of ["formats", "adapt", "prep"]) if (f[k] !== undefined) p[k] = f[k];
  return { type: f.kind, p };
}
function pickSide() {
  const side = $("pickSide"); side.textContent = "";
  const f = PICK.sel;
  if (!f) {
    side.appendChild(el("div", "pick-side-empty", "Pick a filter to see what it does to the frame shown, its settings, and add it. Double-click a card adds it straight away."));
    return;
  }
  const tr = pickTraits(f);
  if (f.kind !== "ai") {                                // a model: no before / after here (it is a GPU job; its engines are what matters)
  const pv = el("div", "pick-prev"); side.appendChild(pv);
    const heavy = tr.gpu || tr.slow || f.status === "args" || f.kind === "my" && false;
    if (!S.info) pv.appendChild(el("div", "pick-prev-msg", "Open a source to see a before / after here."));
    else if (heavy && !PICK.prev[f.pickId]) {
      const b = el("button", null, f.status === "args" ? "Needs settings first - no preview" : "Show the before / after");
      b.disabled = f.status === "args";
      b.title = "This one runs on the GPU or is slow (an AI model builds its TensorRT engine the first time: up to a few minutes)";
      b.onclick = () => pickPreview(f, true); pv.appendChild(el("div", "pick-prev-msg", tr.gpu ? "Runs on the GPU - the first preview can take a while." : "A slow filter.")); pv.appendChild(b);
    } else pickPreview(f, false);
  }
  const h = el("div", "pick-dhead");
  const [g, col] = PICK_ICON[f.kind === "ai" ? "ai" : f.group] || ["•", "#555e74"];
  const ic = el("div", "pick-ico", g); ic.style.background = col;
  const t = el("div"); t.append(el("h3", null, f.name || f.call), el("div", "mono dim", f.kind === "eva" ? "an EVA toolkit stage" : (f.kind === "tpl" || f.kind === "node") ? "Python code" + (f.kind === "node" ? " · your custom filter" : "") : (f.call || "") + (f.plugin ? " · " + f.plugin : "")));
  h.append(ic, t); side.appendChild(h);
  const d = f.kind === "ai" ? "" : pickDesc(f); if (d) side.appendChild(el("div", "pick-ddesc", d));    // a model: its facts are in the engine section
  if (f.note && f.note !== d) side.appendChild(el("div", "pick-ddesc dim", f.note));
  if (f.kind === "ai") { const host = el("div", "pick-eng"); side.appendChild(host); pickEngines(f, host); }
  if (f.kind === "tpl" || f.kind === "node") { const pre = el("pre", "flt-code full tpl"); pre.innerHTML = pyHighlight(f.code); side.appendChild(pre); }
  else if (f.kind === "eva") {
    const ps = (f.block.params || []).filter((q) => !q.show);
    if (ps.length) { const t2 = el("div", "pick-args"); for (const q of ps) t2.append(el("span", null, q.label), el("span", "dim", q.kind === "choice" ? ((q.options || []).find((o) => o[0] === q.default) || [0, q.default])[1] : String(q.default))); side.appendChild(t2); }
  } else if ((f.args || []).length && f.kind !== "ai") {
    const t2 = el("div", "pick-args");
    for (const a of f.args) {
      const clipArg = f.kind === "vs" ? (a.type || "").startsWith("vnode") : /VideoNode/.test(a.type || "");
      const def = (f.defaults && f.defaults[a.name] !== undefined) ? String(f.defaults[a.name])
        : a.auto ? "made automatically" : f.kind === "vs" ? (a.opt ? "default" : "required") : (a.required ? "required" : String(a.default ?? ""));
      const v = el("span", def === "required" ? "req" : clipArg ? "clip" : "dim", def);
      t2.append(el("span", "mono", a.name), v);
    }
    side.appendChild(t2);
  }
  if (f.status === "args") side.appendChild(el("div", "vsf-warn", "You fill in: " + (f.needs || vsfRequired(f).join(", ")) + " - after adding it, in its settings."));
  const spec = pickPreviewSpec(f);                     // what it adds: the code you get to edit (plugins / vs-jetpack / yours)
  if (["vs", "jet", "my"].includes(f.kind)) {
    side.appendChild(el("div", "pick-code-h", "Added as this code - edit its values after adding"));
    const pre = el("pre", "flt-code full tpl"); pre.innerHTML = pyHighlight(filterTemplate(f)); side.appendChild(pre);
  } else if (f.kind !== "tpl" && f.kind !== "node" && f.kind !== "ai") {   // code ones: their code is shown above; a model: its engines
    side.appendChild(el("div", "pick-code-h", "In the script (click to copy)"));
    side.appendChild(filterCodeEl(spec.type, spec.p, f.kind === "eva" || f.kind === "ai" ? f.name : "", true));
  }
  if (f.formats && f.formats.length) side.appendChild(el("div", "dim small", "Works on " + f.formats.slice(0, 6).join(", ") + (f.formats.length > 6 ? " …" : "") + " (other formats are converted for it)."));
  const acts = el("div", "pick-dacts");
  const add = el("button", "primary", VSFB.onAdd ? "Add to the chain" : "Add"); add.onclick = () => pickAdd(f);
  acts.append(add);
  if (f.kind === "node") {                             // your custom filter: delete it from the library
    const del = el("button", "ghost", "Delete"); del.title = "Delete this custom filter from the library (filters already added stay)";
    del.onclick = async () => { if (!confirm(`Delete "${f.name}" from your filters?`)) return; const r = await rpc("my_node_delete", { name: f.name }); PICK.sel = null; vsfNodesSet(r.nodes); };
    acts.append(del);
  }
  side.appendChild(acts);
}
// ---- the engines of an AI model: Automatic (builds the first time) or a built one; details, build a new one, delete
const engShape = (e) => e.dynamic ? `dynamic ${e.dynamic.min[2]}×${e.dynamic.min[3]} – ${e.dynamic.max[2]}×${e.dynamic.max[3]}` : `static ${e.shape ? e.shape.join("×") : "?"}`;
function engSettingsText(r) {
  const s = r.settings || {};
  const shape = r.profile ? `dynamic ${r.profile[0][2]}×${r.profile[0][3]} – ${r.profile[2][2]}×${r.profile[2][3]}` : "static: one engine per picture size";
  return `${s.precision} · ${shape} · level ${s.opt_level} · ${(s.workspace / 1024).toFixed(s.workspace % 1024 ? 1 : 0)} GB workspace`;
}
async function pickEngines(f, host) {
  host.textContent = "Looking for built engines…";
  const w = S.info ? S.info.width : 720, h = S.info ? S.info.height : 480;
  let r;
  try { r = await rpc("ai_engines", { model: f.path, engine: PICK.eng[f.path] || null, w, h }); } catch { host.textContent = "The engines could not be listed (see the log)."; return; }
  if (!PICK.sel || PICK.sel.pickId !== f.pickId) return;                      // another model was picked meanwhile
  host.textContent = "";
  if (r.error) { host.appendChild(el("div", "dim small", r.error)); return; }
  const head = el("div", "pe-sec"); head.append(el("span", null, "ENGINE"), el("i"), el("span", "dim", `TensorRT · ${r.gpu} · ${r.trt}`)); host.appendChild(head);
  const mi = r.model.info || {}, chips0 = el("div", "pe-chips");
  for (const t of [mi.kind === "onnx" ? "ONNX" : `PyTorch${mi.arch ? " · " + mi.arch : ""}`, mi.scale ? `${mi.scale}x` : null, mi.fixed ? `fixed input ${mi.fixed[0]}×${mi.fixed[1]}` : "any input size",
                   mi.vsr ? `video model: ${mi.frames || "?"} frames in` : null, `stored as ${mi.precision || "?"}`, `${r.model.mb} MB`]) if (t) chips0.appendChild(el("span", "bd", t));
  if (r.input) chips0.appendChild(el("span", "bd ok", `for this video: input ${r.input.join("×")}`));
  host.appendChild(chips0);
  if (S.backend && S.backend !== "tensorrt") host.appendChild(el("div", "dim small", "Engines are used with the TensorRT backend (Settings)."));
  const refresh = () => pickEngines(f, host);
  const eng = r.engines, anyCurrent = eng.some((e) => e.current);
  const B = PICK.building;
  if (B && B.path === f.path) {                     // a build asked for here is running: its progress
    const bx = el("div", "pe-build");
    const top = el("div", "pe-build-top");
    const stop = el("button", "pe-stop", "Stop"); stop.title = "Stop the build"; stop.onclick = () => rpc("cancel", {}).catch(() => {});
    top.append(el("b", null, "⚒ Building the engine…"), stop);
    const bar = el("div", "pe-bar big"), fill = el("i"); fill.id = "peBuildFill"; bar.appendChild(fill);
    const txt = el("div", "pe-s"); txt.id = "peBuildText";
    bx.append(top, bar, txt); host.appendChild(bx);
    pickEngineProgressUi();
  }
  const radio = el("span", "pe-r");
  // Automatic
  const auto = el("div", "pe-opt" + (anyCurrent ? "" : " sel"));
  auto.append(el("span", "pe-r"));
  const am = el("div"); am.append(el("b", null, "Automatic"),
    el("div", "pe-s", anyCurrent ? "Uses the engine that matches these settings; builds one if it is missing." : "Builds the engine the first time it is needed (once, up to a few minutes)."));
  const set = el("div", "pe-s"); set.append("Build settings: " + engSettingsText(r) + "  ");
  const edit = el("button", "link", "Edit…"); edit.title = "Precision, static or dynamic, optimisation level, workspace, trtexec command";
  edit.onclick = (e) => { e.stopPropagation(); vsfEnginePending(f.path, f.name, { ...r.settings, profile: undefined }, (e2, build) => pickEngineChosen(f, e2, build, refresh)); };
  set.appendChild(edit); am.appendChild(set);
  auto.append(am, el("span", "bd", "default"));
  auto.onclick = () => { delete PICK.eng[f.path]; refresh(); };
  host.appendChild(auto);
  if (!eng.length) host.appendChild(el("div", "pe-s", "No engine of this model is built yet."));
  // the built ones
  for (const e of eng) {
    const row = el("div", "pe-opt" + (e.current ? " sel" : "") + (e.usable ? "" : " dim"));
    row.append(el("span", "pe-r"));
    const m = el("div");
    m.append(el("b", null, `Built engine · ${e.dynamic ? `${e.dynamic.min[2]}–${e.dynamic.max[2]} (dynamic)` : `${(e.shape ? e.shape[3] - 32 : "?")} × ${(e.shape ? e.shape[2] - 32 : "?")} picture`}`));
    const chips = el("div", "pe-chips");
    const chip = (t, c) => chips.appendChild(el("span", "bd" + (c ? " " + c : ""), t));
    if (e.current) chip("in use", "ac"); else if (e.fits) chip("matches this video", "ok"); else if (e.usable && !e.dynamic) chip("for another picture size");
    if (!e.usable) chip(e.why, "wa");
    chip(e.precision); chip(engShape(e)); chip(`level ${e.opt_level}`); chip(`${e.mb} MB`); chip(`built ${e.built}`);
    if (e.meta && e.meta.seconds) chip(`${Math.floor(e.meta.seconds / 60)} min ${Math.round(e.meta.seconds % 60)} s to build`);
    m.appendChild(chips);
    const btns = el("span", "pe-btns");
    const info = el("button", "pe-ib", "ⓘ"); info.title = "Details";
    info.onclick = (ev) => { ev.stopPropagation(); PICK.engOpen[e.file] = !PICK.engOpen[e.file]; refresh(); };
    const del = el("button", "pe-ib", "🗑"); del.title = "Delete this engine";
    del.onclick = async (ev) => {
      ev.stopPropagation();
      if (e.usable && !confirm(`Delete this engine (${e.mb} MB)?
It is built again the next time it is needed.`)) return;
      await rpc("ai_engine_delete", { files: [e.file] }); refresh();
    };
    btns.append(info, del);
    row.append(m, btns);
    if (e.usable) row.onclick = async () => {
      const p = await rpc("ai_engine_pick", { model: f.path, file: e.file });
      if (p.engine) { PICK.eng[f.path] = p.engine; refresh(); }
      if (!e.dynamic && !e.fits) status(`That engine is for ${e.shape[3] - 32} × ${e.shape[2] - 32} pictures: its settings are used, and each picture size gets the engine that matches it.`);
    };
    host.appendChild(row);
    if (PICK.engOpen[e.file]) {
      const d = el("div", "pe-detail");
      const L = (k, v) => d.append(el("span", "dim", k), el("span", "mono", String(v)));
      L("file", e.file); L("precision", e.precision + (e.precision === "fp32" ? (e.tf32 ? " (TF32 on)" : " (TF32 off)") : "")); L("optimisation level", e.opt_level);
      if (e.meta && e.meta.workspace) L("workspace", `${e.meta.workspace} MB`);
      if (e.dynamic) L("sizes (min / opt / max)", `${e.dynamic.min.join("×")} / ${e.dynamic.opt.join("×")} / ${e.dynamic.max.join("×")}`); else L("input", (e.shape || []).join("×"));
      L("TensorRT", e.trt); L("GPU", e.gpu); L("size", `${e.mb} MB`); L("built", e.built + (e.meta && e.meta.seconds ? ` · ${e.meta.seconds} s` : " · build time not recorded"));
      host.appendChild(d);
    }
  }
  // actions
  const acts = el("div", "pe-acts");
  const build = el("button", null, "⚒ Build a new engine…");
  build.disabled = !!(B && B.path === f.path);
  build.onclick = () => vsfEnginePending(f.path, f.name, { ...r.settings, profile: undefined }, (e2, doBuild) => pickEngineChosen(f, e2, doBuild, refresh, true));
  acts.appendChild(build);
  const bad = eng.filter((e) => !e.usable);
  if (bad.length) {
    const b2 = el("button", null, `Delete the ${bad.length} that cannot be used`);
    b2.onclick = async () => { if (confirm(`Delete ${bad.length} engine${bad.length > 1 ? "s" : ""} this app cannot use (${bad.reduce((a, e) => a + e.mb, 0).toFixed(0)} MB)?`)) { await rpc("ai_engine_delete", { files: bad.map((e) => e.file) }); refresh(); } };
    acts.appendChild(b2);
  }
  host.appendChild(acts);
  const disk = el("div", "pe-disk"); const bar = el("div", "pe-bar"), fill = el("i"); fill.style.width = Math.min(100, 100 * r.total_mb / Math.max(1, r.folder_mb)) + "%"; bar.appendChild(fill);
  disk.append(bar, el("span", "dim", `${eng.length} engine${eng.length === 1 ? "" : "s"} for this model · ${r.total_mb} MB of ${(r.folder_mb / 1000).toFixed(1)} GB in engines\\`));
  host.appendChild(disk);
}
async function pickEngineChosen(f, e2, doBuild, refresh, force) {     // the engine window was saved (and maybe "Save & build now")
  PICK.eng[f.path] = e2;
  if (!doBuild) { refresh(); return; }
  const w = (S.info && S.info.width) || 720, h = (S.info && S.info.height) || 480;
  PICK.building = { path: f.path, frac: 0, phase: "Starting", t0: Date.now(), n: 1 };
  PICK.building.timer = setInterval(pickEngineProgressUi, 1000);
  refresh();
  status(`Building the engine of ${f.name} (${w}x${h}) - see the bar in the library`);
  try {
    const r = await rpc("build_ai_engine", { model: f.path, engine: e2, w, h, backend: S.backend });
    if (r && r.error) { status(r.error, "err"); log(r.error, "err"); } else status(`Engine ready: ${f.name}`, "ok");
  } catch { /* logged */ }
  clearInterval(PICK.building.timer); PICK.building = null;
  refresh();
}
// "engine_build" events from the engine (TensorRT's progress monitor): the bar in the model's pane
function pickEngineProgress(m) {
  const B = PICK.building; if (!B) return;
  B.frac = Math.max(B.frac || 0, m.frac || 0); B.phase = m.phase || B.phase; B.n = m.build || B.n;
  pickEngineProgressUi();
}
function pickEngineProgressUi() {
  const B = PICK.building, fill = document.getElementById("peBuildFill"), txt = document.getElementById("peBuildText");
  if (!B || !fill || !txt) return;
  const sec = Math.round((Date.now() - B.t0) / 1000);
  fill.style.width = Math.round(100 * (B.frac || 0)) + "%";
  txt.textContent = `${Math.round(100 * (B.frac || 0))} %  ·  ${B.phase}  ·  ${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}` + (B.n > 1 ? `  ·  engine ${B.n} (the model is also checked against another precision)` : "");
}
function pickPreview(f, force) {
  const pv = $("pickSide").querySelector(".pick-prev");
  const key = `${f.pickId}#${S.current}#${S.frame}`;
  const show = (r) => {
    pv.textContent = "";
    if (r.error) { pv.appendChild(el("div", "pick-prev-msg err", "No preview: " + r.error)); return; }
    const a = el("img"), b = el("img"); a.src = "data:image/jpeg;base64," + r.before; b.src = "data:image/jpeg;base64," + r.after;
    b.className = "after"; b.style.clipPath = `inset(0 0 0 ${PICK.split}%)`;
    const line = el("i", "pick-prev-line"); line.style.left = PICK.split + "%";
    pv.append(a, b, line, el("span", "pick-tag l", "before"), el("span", "pick-tag r", `after · frame ${r.frame}${r.size ? ` · ${r.size[0]}×${r.size[1]}` : ""}`));
    pv.onpointermove = (e) => { const rc = pv.getBoundingClientRect(); PICK.split = Math.max(0, Math.min(100, 100 * (e.clientX - rc.left) / rc.width)); b.style.clipPath = `inset(0 0 0 ${PICK.split}%)`; line.style.left = PICK.split + "%"; };
  };
  if (PICK.prev[key]) { show(PICK.prev[key]); return; }
  pv.textContent = ""; pv.appendChild(el("div", "pick-prev-msg", "Rendering the before / after…"));
  clearTimeout(PICK.prevTimer);
  const seq = ++PICK.prevSeq;
  PICK.prevTimer = setTimeout(async () => {
    try {
      const r = await rpc("filter_preview", { state: backendState([S.current]), episode: S.current, frame: S.frame || 0, ...pickPreviewSpec(f) });
      PICK.prev[key] = r; PICK.prev[f.pickId] = true;
      if (seq === PICK.prevSeq && PICK.sel && PICK.sel.pickId === f.pickId) show(r);
    } catch (e) { if (seq === PICK.prevSeq) { pv.textContent = ""; pv.appendChild(el("div", "pick-prev-msg err", "No preview")); } }
  }, force ? 0 : 350);
}
function pickAdd(f) {
  pickSave("pickRecent", [f.pickId, ...pickStore("pickRecent", []).filter((x) => x !== f.pickId)].slice(0, 12));
  const fn = VSFB.onAdd, g = VSFB.g;
  VSFB.fresh = null;                                    // something is added: the new group stays
  vsfBrowserClose();
  if (fn) { fn(f); return; }
  if (f.kind === "eva" || f.kind === "tpl" || f.kind === "node") { vsfAddBlock(g, f); return; }
  vsfAddTo(g, f);
}
// ---- a filter's script text: the VapourSynth line(s) it adds (shown in the library and in each group)
function pyVal(v, raw) {                              // a setting as Python: plugin values are typed as text in the form
  if (v === true || v === false) return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  const t = String(v).trim();
  if (raw) return t;                                   // vs-jetpack: the field is already Python
  if (/^-?\d+(\.\d+)?$/.test(t)) return t;
  if (/^(true|false)$/i.test(t)) return t[0].toUpperCase() + t.slice(1).toLowerCase();
  if (/^-?[\d.]+(\s*,\s*-?[\d.]+)+$/.test(t)) return "[" + t.split(",").map((x) => x.trim()).join(", ") + "]";
  if (t.startsWith("@")) return t.slice(1);            // @group / @prev / @self: the clips the app passes in
  return "'" + t.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'";
}
function filterCode(type, p, label) {
  p = p || {};
  const args = Object.entries(p.args || {}).filter(([, v]) => v !== "" && v != null);
  if (type === "code") return (p.code || (p.args && p.args.code) || "clip = clip").trim();
  if (type === "vs") return `clip = core.${p.call}(clip${args.map(([k, v]) => `, ${k}=${pyVal(v)}`).join("")})`;
  if (type === "jet") { const mod = p.call.split(".")[0]; return `import ${mod}\nclip = ${p.call}(clip${args.map(([k, v]) => `, ${k}=${pyVal(v, true)}`).join("")})`; }
  const dict = (o) => "{" + Object.entries(o).map(([k, v]) => `'${k}': ${typeof v === "object" && v ? JSON.stringify(v).replace(/"/g, "'") : pyVal(v)}`).join(", ") + "}";
  if (type === "ai" || type === "my")
    return `clip = dr.block(clip, job, '${type}', {'call': '${p.call}', 'args': ${dict(p.args || {})}}, ctx)` + (label ? `   # ${label}` : "");
  return `clip = dr.block(clip, job, '${type}', ${dict(p)}, ctx)` + (label ? `   # ${label}` : "");
}
// ---- a filter as an editable code template: its settings as variables with their defaults and what they take,
// then the call using them (and the format it needs) - what a filter becomes when you add it
const PICK_DOCS = { vs_tiletools: "https://github.com/pifroggi/vs_tiletools", vs_undistort: "https://github.com/pifroggi/vs_undistort",
                    vsdenoise: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack", vsdeband: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack",
                    vstools: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack", vsmasktools: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack",
                    vsaa: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack", vsdehalo: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack",
                    vsdeinterlace: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack", vsscale: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack",
                    vsrgtools: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack", vskernels: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack",
                    vsexprtools: "https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack" };
function wrapComment(text, width = 96) {
  const out = []; let line = "#";
  for (const w of String(text).replace(/\s+/g, " ").trim().split(" ")) {
    if ((line + " " + w).length > width) { out.push(line); line = "#"; }
    line += " " + w;
  }
  if (line !== "#") out.push(line);
  return out;
}
function filterTemplate(f) {
  const jet = f.kind === "jet", my = f.kind === "my", mod = (f.call || "").split(".")[0];
  const L = [`# ${f.name || f.call}  -  ${jet ? f.call + " (vs-jetpack style)" : my ? f.call + " (your filter)" : "core." + f.call + " (plugin " + (f.plugin || mod) + ")"}`];
  if (PICK_DOCS[mod]) L.push(`# Docs: ${PICK_DOCS[mod]}`);
  const d = (f.doc || "").trim(); if (d) L.push(...wrapComment(d));
  if (f.note && f.note !== d) L.push(...wrapComment(f.note));
  L.push("");
  const all = f.args || [];
  const isClip = (a) => (jet ? /VideoNode/.test(a.type || "") : (a.type || "").startsWith("vnode"));
  // the app's shortcuts written out as code: picture size, NTSC / PAL, "this clip"
  const TOK = { "@w": "clip.width", "@h": "clip.height", "@std": "('pal' if dr.is_pal(job) else 'ntsc')", "@self": "clip", "@group": "clip", "@prev": "clip",
                w: "clip.width", h: "clip.height", current: "clip", group: "clip", prev: "clip" };
  const rawDefault = (a) => (f.defaults && f.defaults[a.name] !== undefined ? String(f.defaults[a.name]) : jet ? (a.default ?? "") : "");
  const tokenOf = (a) => { const v = rawDefault(a).trim(); return TOK[v] || null; };
  // when the app's own runner is needed: helper clips it makes (MVTools super / vectors, AWarp's mask), a step it runs
  // first (scene detection), or a filter that only takes grey pictures (the runner puts the colour back afterwards)
  const grayOnly = f.formats && f.formats.length && f.formats.every((x) => /^GRAY/.test(x));
  const viaApp = (!jet && !my && (all.some((a) => a.auto) || grayOnly)) || !!f.prep;
  const args = all.filter((a) => !a.auto);
  const plain = (v) => /^(None|True|False|-?\d+(\.\d+)?(e-?\d+)?|'[^'<>]*'|"[^"<>]*"|\[[^<>]*\]|\([^<>]*\))$/.test(String(v).trim());
  const val = (a) => {
    if (viaApp && /^@/.test(rawDefault(a).trim())) return `'${rawDefault(a).trim()}'`;   // the app's runner reads its own shortcuts
    const t = tokenOf(a); if (t) return t;
    if (f.defaults && f.defaults[a.name] !== undefined) return jet ? String(f.defaults[a.name]) : pyVal(f.defaults[a.name]);
    if (jet || my) return a.default !== "" && a.default != null ? String(a.default) : "None";
    return "None";
  };
  const kept = jet ? args.filter((a) => !tokenOf(a) && !(f.defaults && f.defaults[a.name] !== undefined) && a.default !== "" && a.default != null && !plain(a.default)) : [];
  const used = args.filter((a) => !kept.includes(a));
  const hint = (a) => {
    if (tokenOf(a)) return { "clip.width": "the picture's width", "clip.height": "the picture's height", clip: "this clip" }[tokenOf(a)] || "NTSC or PAL";
    const t = String(a.type || "").replace(/vapoursynth\./g, "").replace(/<class '([^']+)'>/g, "$1");
    const req = jet || my ? a.required : !a.opt;
    return (isClip(a) ? "a clip (clip = the picture here)" : t.slice(0, 48)) + (req ? " · needed" : jet || my ? "" : " · None = the plugin's own default");
  };
  const w = Math.max(8, ...used.map((a) => a.name.length));
  const lines = used.map((a) => [`${a.name.padEnd(w)} = ${val(a)}`, hint(a)]);
  const cw = Math.max(0, ...lines.map((l) => l[0].length)) + 2;
  for (const [code, h] of lines) L.push(`${code.padEnd(cw)}# ${h}`);
  if (kept.length) {
    L.push("# kept at their defaults (not plain values - pass your own in the call below if you need to):");
    const short = (v) => String(v).replace(/ object at 0x[0-9A-Fa-f]+/g, "").replace(/<class '([^']+)'>/g, "$1").replace(/<(\w+\.\w+): [^>]*>/g, "$1")
      .replace(/^<|>$/g, "").replace(/\b[a-z_]+\.(?:[a-z_]+\.)*([A-Z]\w*)/g, "$1").slice(0, 70);
    for (const a of kept) L.push(`#   ${a.name} = ${short(a.default)}`);
  }
  if (args.length) L.push("");
  if (jet) L.push(`import ${mod}`);
  if (viaApp || my) {                                  // through the app's runner (it makes what the filter needs)
    const extra = ["formats", "adapt", "prep"].filter((k) => f[k] !== undefined).map((k) => `, '${k}': ${JSON.stringify(f[k]).replace(/"/g, "'")}`).join("");
    const why = all.some((a) => a.auto) ? "the app makes its helper clips (" + all.filter((a) => a.auto).map((a) => a.name).join(", ") + ")"
      : f.prep ? "the app marks scene changes first" : grayOnly ? "it works on the brightness; the app keeps the colour" : "your filter, run by the app";
    L.push(`# run by the app: ${why}`);
    L.push(`clip = dr.block(clip, job, '${f.kind}', {'call': '${f.call}', 'args': {${used.map((a) => `'${a.name}': ${a.name}`).join(", ")}}${extra}}, ctx)`);
    return L.join("\n");
  }
  // the format it works in (the app turns the result back into the chain's own format afterwards)
  const usual = ["YUV420P16", "YUV444P16"];
  const need = f.adapt || (f.formats && f.formats.length && !usual.every((x) => f.formats.includes(x)) ? f.formats[0] : null);
  if (need) {
    if (/^RGB/.test(need)) L.push(`clip = core.resize.Bilinear(clip, format=vs.${need}, matrix_in_s="170m")   # it works in ${need}`);
    else L.push(`clip = core.resize.Bilinear(clip, format=vs.${need})   # it works in ${need}`);
  }
  const pos = used.filter((a) => a.pos), kw = used.filter((a) => !a.pos);
  const callArgs = ["clip", ...pos.map((a) => a.name), ...kw.map((a) => `${a.name}=${a.name}`)].join(", ");
  L.push(`clip = ${jet ? f.call : "core." + f.call}(${callArgs})`);
  if (need) L.push("# the app turns the result back into the chain's own format by itself");
  return L.join("\n");
}
// a small code editor: coloured Python behind a transparent text area
function codeEditor(value, onChange, numbers) {     // numbers: line numbers on the left
  const box = el("div", "cedit" + (numbers ? " nums" : "")), pre = el("pre", "cedit-hl"), ta = el("textarea", "cedit-ta");
  const gut = el("div", "cedit-gut"), area = el("div", "cedit-area");
  ta.spellcheck = false; ta.wrap = "off"; ta.value = value;
  const paint = () => {
    pre.innerHTML = pyHighlight(ta.value) + "\n";
    const n = ta.value.split("\n").length; ta.style.height = Math.min(560, Math.max(80, n * 19 + 20)) + "px";   // tall enough for the whole template
    pre.style.height = ta.style.height;
    if (numbers) { gut.textContent = Array.from({ length: n }, (_, k) => k + 1).join("\n"); gut.style.height = ta.style.height; gut.scrollTop = ta.scrollTop; }
  };
  ta.addEventListener("input", paint);
  ta.addEventListener("scroll", () => { pre.scrollTop = ta.scrollTop; pre.scrollLeft = ta.scrollLeft; gut.scrollTop = ta.scrollTop; });
  ta.addEventListener("change", () => onChange(ta.value));
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Tab") { e.preventDefault(); const a = ta.selectionStart; ta.setRangeText("    ", a, ta.selectionEnd, "end"); paint(); }
    if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); onChange(ta.value); }
  });
  if (numbers) { area.append(pre, ta); box.append(gut, area); } else box.append(pre, ta);
  setTimeout(paint, 0);
  return box;
}
function filterCodeEl(type, p, label, full) {         // the code, copyable (click)
  const code = filterCode(type, p, label), box = el(full ? "pre" : "div", full ? "flt-code full" : "flt-code", full ? code : code.split("\n").pop());
  box.title = (full ? "" : code + "\n\n") + "Click to copy";
  box.onclick = (e) => { e.stopPropagation(); navigator.clipboard.writeText(code).then(() => status("Copied: " + code.split("\n").pop(), "ok")).catch(() => {}); };
  return box;
}
function blockForm(type, p, onChange) {           // the settings of an app stage (blocks.py params) as real controls
  const T = CHM.lib && (CHM.lib.blocks || []).find((b) => b.id === type), box = el("div", "blk-form");
  if (!T) { box.appendChild(el("div", "vsf-doc", "Loading…")); chainLibrary().then(() => vsgChanged()).catch(() => {}); return box; }
  const shown = (q) => { if (!q.show) return true; const m = /^(\w+)([=!])(.*)$/.exec(q.show); if (!m) return true;
    const dq = T.params.find((x) => x.key === m[1]); const v = String(p[m[1]] ?? (dq ? dq.default : "")); const hit = v === m[3] || m[3].split("|").includes(v); return m[2] === "=" ? hit : !hit; };
  for (const q of T.params) {
    if (!shown(q)) continue;
    const v = p[q.key] ?? q.default, row = el("div", "field"), lab = el("label", null, q.label); if (q.tip) lab.title = q.tip;
    let inp;
    if (q.kind === "choice") {
      if ((q.options || []).length <= 3) {
        inp = el("div", "seg"); for (const [k2, t] of q.options) { const b = el("button", v === k2 ? "on" : "", t); b.onclick = () => { p[q.key] = k2; onChange(); }; inp.appendChild(b); }
      } else { inp = el("select"); for (const [k2, t] of q.options || []) { const o = el("option", null, t); o.value = k2; inp.appendChild(o); } inp.value = v; inp.onchange = () => { p[q.key] = inp.value; onChange(); }; }
    } else if (q.kind === "bool") { inp = el("input"); inp.type = "checkbox"; inp.checked = !!v; inp.onchange = () => { p[q.key] = inp.checked; onChange(); }; }
    else if (q.kind === "code") { inp = el("textarea", "vsf-code"); inp.rows = 6; inp.value = v; inp.onchange = () => { p[q.key] = inp.value; onChange(); }; }
    else { inp = el("input"); inp.type = "number"; inp.value = v; if (q.min != null) inp.min = q.min; if (q.max != null) inp.max = q.max; if (q.step != null) inp.step = q.step;
           inp.onchange = () => { const n = parseFloat(inp.value); if (!isNaN(n)) { p[q.key] = q.kind === "int" ? Math.round(n) : n; onChange(); } }; }
    row.append(lab, inp); box.appendChild(row);
  }
  if (!T.params.length) box.appendChild(el("div", "vsf-doc", "No settings."));
  return box;
}
function vsfBrowserClose() {
  $("vsfDlg").classList.add("hidden"); VSFB.onAdd = null; PICK.only = null;
  const fresh = VSFB.fresh; VSFB.fresh = null;             // a group made just for this and nothing added: gone again
  if (fresh) {
    const g = S.vs_groups.find((x) => x.id === fresh);
    if (g && !g.filters.length) { S.vs_groups = S.vs_groups.filter((x) => x !== g); vsgChanged(); }
  }
}
async function vsfBrowser(g, onAdd, opts = {}) {
  VSFB.g = g; VSFB.sel = null; VSFB.onAdd = onAdd || null; VSFB.fresh = opts.fresh && g ? g.id : null;
  PICK.only = opts.only || null;
  PICK.sel = null; PICK.cat = PICK.only || opts.cat || "all"; PICK.chip = "all"; VSFB.cat = PICK.cat;
  $("pickTitle").textContent = PICK.only === "ai" ? "Add a model" : "Add a filter";
  $("vsfDlgSearch").placeholder = PICK.only === "ai" ? "Search your models by name…" : "Search by name, category or what it does: denoise, dehalo, deband, IVTC, sharpen, mask…";
  $("vsfDlgFor").textContent = opts.where || (g ? "→ " + vsfWhere(g) : "");
  $("vsfDlg").classList.remove("hidden");
  $("vsfDlgSearch").value = ""; $("vsfDlgSearch").focus();
  pickBuild(); pickRender();
  try { await Promise.all([vsfLibrary(), chainLibrary()]); pickBuild(); pickRender(); }
  catch { $("vsfDlgRows").textContent = "The filter list could not be loaded (see the log)."; }
}
function vsfBrowserRender() { pickBuild(); pickRender(); }        // the AI / your-filters bars call this after changes
function pickMove(d) {
  const cards = [...$$("#vsfDlgRows .pick-card")];
  if (!cards.length) return;
  let i = cards.findIndex((c) => c.classList.contains("sel"));
  i = Math.max(0, Math.min(cards.length - 1, i < 0 ? 0 : i + d));
  cards[i].click(); cards[i].scrollIntoView({ block: "nearest" });
}
document.addEventListener("DOMContentLoaded", () => {
  $("vsfDlgClose").onclick = vsfBrowserClose;
  $("vsfDlg").onclick = (e) => { if (e.target === $("vsfDlg")) vsfBrowserClose(); };
  $("vsfDlgSearch").oninput = () => { pickRender(); };
  $("vsfDlgSearch").onkeydown = (e) => {
    if (e.key === "Escape") { e.preventDefault(); vsfBrowserClose(); }
    else if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); pickMove(1); }
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); pickMove(-1); }
    else if (e.key === "Enter" && PICK.sel) { e.preventDefault(); pickAdd(PICK.sel); }
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("vsfDlg").classList.contains("hidden")) vsfBrowserClose(); });
});
