// Repair modes: Guided (the step panels) | Chain (your own order of blocks) | Script (a VapourSynth script).
// Chain = S.chain: [{uid, type, on, p, label, doc, argspec}] run by blocks.build_chain in that order. Every repair stage
// of the app, every VapourSynth plugin / vs-jetpack function, your AI models and filters, and Python code are blocks.
// "View" on a block previews the picture after it (S.chainView -> job["chain_view"], preview only).
"use strict";

const CHM = { lib: null, open: null, drag: null, addAt: null };
const CHAIN_COL = { Source: "#555e74", Telecine: "#3b5378", Cleanup: "#2f8f86", Temporal: "#a0773a", "Crop & size": "#6d6a9e", Colour: "#a05a8a",
                    Utility: "#7a6a2c", "JET tools": "#7a6a2c", Output: "#555e74", VapourSynth: "#7a6a2c", AI: "#b4566a", Yours: "#b4566a" };
const CHAIN_TEMPLATES = [
  { name: "Mask merge (two ways, mixed through a mask)", desc: "Denoise strongly, but keep the outlines from before: a typical branch + mask.",
    code: "# two ways from the same picture, mixed through a mask (white = the first, black = the second)\nstrong = core.zsmooth.FluxSmoothST(clip, temporal_threshold=12, spatial_threshold=12)   # way 1: strong denoise\nmask = vsmasktools.Kirsch.edgemask(vstools.get_y(clip)).std.Maximum()   # the outlines\nclip = core.std.MaskedMerge(strong, clip, mask, first_plane=True)" },
  { name: "Only some frames (a frame range)", desc: "A filter on frames A to B only (source frame numbers, as on the Timeline).",
    code: "# a filter on some frames only: the picture from frame A to B (source frames) comes from `alt`\nalt = core.std.BoxBlur(clip, hradius=2, vradius=2)            # your filter here\nclip = dr.splice_src(clip, alt, [(1000, 1500)])" },
  { name: "Deband", desc: "Smooth banding in flat gradients (vs-jetpack).", code: "clip = vsdeband.f3k_deband(clip, radius=16, thr=24)" },
];
function chainUid() { return "b" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
async function chainLibrary() {
  if (!CHM.lib) { const r = await rpc("block_library", {}); CHM.lib = r; }
  return CHM.lib;
}
function chainSchema(type) { return CHM.lib && (CHM.lib.blocks || []).find((b) => b.id === type); }
function chainName(b) {
  if (b.label) return b.label;
  const T = chainSchema(b.type);
  if (b.type === "code") return "Custom code";
  return T ? T.name : (b.p && b.p.call) || b.type;
}
function chainCat(b) { const T = chainSchema(b.type); return T ? T.cat : b.type === "ai" ? "AI" : b.type === "my" ? "Yours" : "VapourSynth"; }
function chainSummary(b) {
  const T = chainSchema(b.type), p = b.p || {};
  if (b.type === "code") return (p.code || (p.args && p.args.code) || "").split("\n").find((l) => l.trim() && !l.trim().startsWith("#")) || "Python";
  if (!T || !T.params.length) return (p.call || "") + (p.args ? " " + Object.entries(p.args).filter(([, v]) => v !== "" && v != null).map(([k, v]) => `${k}=${v}`).slice(0, 3).join(" ") : "");
  return T.params.filter((q) => chainShown(q, b)).slice(0, 4).map((q) => {
    const v = p[q.key] ?? q.default;
    if (q.kind === "choice") { const o = (q.options || []).find((x) => x[0] === v); return o ? o[1] : String(v); }
    if (q.kind === "bool") return v ? q.label : null;
    return `${q.label} ${v}`;
  }).filter(Boolean).join(" · ");
}
function chainShown(q, b) {
  if (!q.show) return true;
  const m = /^(\w+)([=!])(.*)$/.exec(q.show);
  if (!m) return true;
  const T = chainSchema(b.type), dq = T && T.params.find((x) => x.key === m[1]);
  const v = String((b.p || {})[m[1]] ?? (dq ? dq.default : ""));
  const hit = v === m[3] || m[3].split("|").includes(v);
  return m[2] === "=" ? hit : !hit;
}
function chainChanged(preview = true) {
  S.chain = S.chain.map((b) => b);                 // a new array: undo / history see a change
  CHM.lastChain = S.chain;
  changed("chain", S.chain);
  if (!preview) return;
  chainRender();
}

// ---------------------------------------------------------------- the Repair panel: Guided + the script
// The Guided panels as they were, with the script of their settings beside them when you want it ("</> Script").
// "Edit" in the script turns it into a text you type (Script mode); "Back to the panels" leaves it.
function repairMode() { return S.script ? "script" : "guided"; }     // "script" = you typed / loaded your own script
async function setRepairMode(m) { scriptWinShow(m === "script"); }     // the script is a window (scriptmode.js), not a mode of the panel
function rmodeUi() {                                  // the panels are always shown; the old side pane and Script mode are gone
  document.body.classList.remove("rmode-chain", "rmode-script", "live-side");
  document.body.classList.add("live-off");
  $("chainPanel").classList.add("hidden");
  $("scriptPanel").classList.add("hidden");
  $("liveScript").classList.add("hidden");
  guidedBindSelect();
  setTimeout(layoutImage, 0);
}
function rmodeSync() { guidedBindSelect(); }
function guidedBindSelect() {                         // clicking a panel shows its lines in the script window
  for (const g of $$('.insp [data-panel="adjust"] > .group')) {
    if (g._selBound) continue;
    g._selBound = true;
    g.addEventListener("pointerdown", () => {
      CHM.sel = g;
      for (const x of $$('.insp [data-panel="adjust"] > .group')) x.classList.toggle("chain-sel", x === g);
      if (typeof scriptWinReveal === "function") scriptWinReveal(g);
    }, true);
  }
}
function liveShown() { scriptWinShow(!scriptWinOpen()); }
function initRepairModes() {
  S.rmode = "guided";
  $("btnScriptSide").onclick = () => scriptWinShow(!scriptWinOpen());
  $("btnAnalyzeAll").onclick = analyzeAll;
  $("btnRepairNext").onclick = () => goto("colour");
  rmodeUi();
}

// ---------------------------------------------------------------- one list: plain names, on / off, what was found
// main = the settings shown when a step is open (the rest is behind More); off / on = what the switch sets
const STEPS = {
  telecine: { name: "Telecine & interlacing", main: ["scan", "mode", "order"] },
  composite: { name: "Dots & rainbows", main: ["ai_model", "crawl", "rainbow"],
    isOn: () => [S.crawl, S.rainbow, S.composite, S.ai_model].some((v) => v && v !== "off"),
    off: { crawl: "off", rainbow: "off", composite: "off", ai_model: "off" }, on: { ai_model: "precise", crawl: "normal", rainbow: "normal" } },
  compression: { name: "MPEG blocks", main: ["deblock"], isOn: () => S.deblock && S.deblock !== "off", off: { deblock: "off" }, on: { deblock: "normal" } },
  halos: { name: "Halos", main: ["ghost", "halo_method", "ghost_strength"], isOn: () => S.ghost && S.ghost !== "off", off: { ghost: "off" }, on: { ghost: "auto" } },
  colour: { name: "Colour fix", main: ["sx", "sy", "bleed"],
    isOn: () => !String(S.bleed || "off").startsWith("off") || +S.sx || +S.sy || !!S.neutral,
    off: { bleed: "off|normal", sx: 0, sy: 0, neutral: false }, on: { bleed: "ai|normal", neutral: true } },
  crop: { name: "Crop", main: ["crop.l", "crop.r", "crop.t", "crop.b"], isOn: () => Object.values(S.crop || {}).some((v) => +v),
    off: { crop: { l: 0, r: 0, t: 0, b: 0 } }, on: {} },
  stab: { name: "Stabilizer", main: ["stab", "stab_level"], isOn: () => S.stab && S.stab !== "off", off: { stab: "off" }, on: { stab: "ustab" } },
  size: { name: "Size", main: ["aspect", "size"] },
  sharpen: { name: "Sharpen & lines", main: ["sharp.luma", "sharp.lines"], isOn: () => S.sharp && (+S.sharp.luma || +S.sharp.chroma || +S.sharp.lines),
    off: { sharp: { luma: 0, chroma: 0, lines: 0 } }, on: { sharp: { luma: 0.5 } } },
};
const FOUND_GID = { frames: "telecine", composite: "composite", block: "compression", ghost: "halos", chroma: "colour", stab: "stab", crop: "crop" };
function repairFound(kind, text) {                     // a short "found: ..." line for the step an Auto / Analyze measured
  const gid = FOUND_GID[kind];
  if (!gid || !text) return;
  let t = String(text).replace(/^[^:]{0,40}:\s*/, "").replace(/\s*\(suggestion.*$/, "");
  t = t.split(/(?<=[.;])\s|\s-\s/)[0].replace(/[.;]$/, "");
  t = t.replace(/\s*\([^)]*(\)|$)/g, "").replace(/\s{2,}/g, " ").trim();     // the measurements in brackets: in the log, not here
  S.found = S.found || {};
  S.found[gid] = t.length > 70 ? t.slice(0, 68) + "…" : t;
  simpleDecorate();
}
function stepSwitch(gid, on) {
  const st = STEPS[gid];
  S.stepSaved = S.stepSaved || {};
  if (!on) {                                           // remember what it was, so On brings exactly that back
    S.stepSaved[gid] = {};
    for (const k of Object.keys(st.off)) S.stepSaved[gid][k] = structuredClone(S[k]);
    for (const [k, v] of Object.entries(st.off)) changed(k, structuredClone(v));
  } else {
    const back = S.stepSaved[gid];
    const vals = back && Object.keys(back).length ? back : st.on;
    for (const [k, v] of Object.entries(vals)) changed(k, k === "sharp" ? { ...(S.sharp || {}), ...v } : structuredClone(v));
    delete S.stepSaved[gid];
  }
  simpleDecorate();
}
function simpleDecorate() {
  const panel = document.querySelector('.insp [data-panel="adjust"]');
  if (!panel) return;
  for (const g of panel.querySelectorAll(":scope > .group[data-gid]")) {
    const gid = g.dataset.gid, st = STEPS[gid], head = g.querySelector(".group-head");
    if (!head) continue;
    if (st) {
      const h3 = head.querySelector("h3");
      if (h3 && h3.textContent !== st.name) h3.textContent = st.name;
      // the main settings stay, the rest is "more"
      for (const f of g.querySelectorAll(".field")) {
        if (f.closest(".rng-box")) continue;
        const b = f.querySelector("[data-bind]"), key = b ? b.dataset.bind : null;
        f.classList.toggle("adv-field", !key || !st.main.includes(key));
      }
      if (!g.querySelector(":scope > .more-btn")) {
        const mb = el("button", "link more-btn", "More ▾");
        mb.onclick = (e) => { e.stopPropagation(); g.classList.toggle("more-open"); mb.textContent = g.classList.contains("more-open") ? "Less ▴" : "More ▾"; };
        g.appendChild(mb);
      }
      // on / off
      let sw = head.querySelector(":scope .step-sw");
      if (st.isOn) {
        if (!sw) {
          sw = el("button", "step-sw"); sw.setAttribute("aria-label", "On / off");
          sw.addEventListener("click", (e) => { e.stopPropagation(); stepSwitch(gid, !st.isOn()); });
          const at = head.querySelector(".gcaret") || null;
          head.appendChild(sw);
          if (at) head.appendChild(at);
        }
        const on = !!st.isOn();
        sw.classList.toggle("on", on); sw.title = on ? "On - click to switch this step off" : "Off - click to switch it on";
        g.classList.toggle("step-off", !on);
      }
    }
    // what Analyze found
    let fd = head.querySelector(".found");
    const txt = S.found && S.found[gid];
    if (txt) {
      if (!fd) { fd = el("div", "found"); const sum = head.querySelector(".gsum"); (sum ? sum.parentNode : head).appendChild(fd); }
      fd.textContent = "found: " + txt;
    } else if (fd) fd.remove();
  }
}
// ---------------------------------------------------------------- one Analyze for everything
const ANALYZE_ALL = [["Telecine", null], ["Dots & rainbows", "composite"], ["MPEG blocks", "block"], ["Halos", "ghost"],
                     ["Colour fix", "chroma"], ["Crop", "crop"], ["Stabilizer", "stab"]];
async function analyzeAll() {
  if (!S.info || S.busy || S.analyzing) return;
  S.analyzing = true;
  const btn = $("btnAnalyzeAll"), bar = $("analyzeBar");
  bar.classList.remove("hidden");
  const quiet = { classList: { add() {}, remove() {} }, textContent: "" };
  try {
    for (let i = 0; i < ANALYZE_ALL.length; i++) {
      const [name, kind] = ANALYZE_ALL[i];
      btn.textContent = `Analyzing ${i + 1} / ${ANALYZE_ALL.length} · ${name}…`;
      bar.firstElementChild.style.width = (100 * i / ANALYZE_ALL.length) + "%";
      if (!kind) await telAnalyze(true);
      else if (kind === "crop") await autoCrop();
      else await autoGroup(kind, quiet);
      if (kind === "composite" && (S.ai_model === "field" || S.ai_model === "uni")) { await autoGroup("block", quiet); await autoGroup("ghost", quiet); i += 2; }
    }
    bar.firstElementChild.style.width = "100%";
    status("Analyzed: every step is set from what was found - check the picture (Compare) and change what does not look right.", "ok");
  } catch { /* logged */ }
  S.analyzing = false;
  btn.textContent = "Analyzed ✓ · analyze again";
  setTimeout(() => bar.classList.add("hidden"), 800);
  simpleDecorate();
}

// ---------------------------------------------------------------- Chain: the Guided panel as a chain
const CHAIN_LINES = {                                  // which script lines belong to which block
  source: /dr\.source\(/, telecine: /ivtc_|ntsc_bob|pal_process|deblend|anti_alias|vs_filters\(clip, job, "(ivtc|after_ivtc)"\)/,
  composite: /xtalk|composite|cleanup|stablenet|uninet|ai_clean|vs_filters\(clip, job, "after_composite"\)/, compression: /deblock/,
  halos: /halo|ghost/, colour: /colour_fix|chroma_fix|repair_colour|neutral/, crop: /dr\.crop\(/, stab: /stabil|still_lock/,
  size: /resize_output|upscale|dr\.size/, sharpen: /sharpen|line_dark/, output: /set_output|output_format/,
};
const CHAIN_BAR = { telecine: "#3b5378", composite: "#a0773a", compression: "#6d6a9e", halos: "#2f8f86", colour: "#2f8f86", crop: "#555e74",
                    stab: "#a0773a", size: "#555e74", sharpen: "#a0773a" };
function chainDecorate() {
  const panel = document.querySelector('.insp [data-panel="adjust"]');
  if (!panel) return;
  for (const x of panel.querySelectorAll(":scope > .chain-add")) x.remove();
  const blocks = [...panel.querySelectorAll(":scope > .group, :scope > .chain-pin")].filter((g) => !g.classList.contains("hidden") && g.offsetParent !== null);
  let n = 0;
  for (const g of blocks) {
    g.dataset.n = ++n;
    const num = g.querySelector(":scope > .chain-n");     // no numbers in the list any more
    if (num) num.remove();
    const gid = g.dataset.gid || g.dataset.pin || "";
    g.style.setProperty("--bar", g.classList.contains("vsf-group") ? "#7a6a2c" : CHAIN_BAR[gid] || "#555e74");
    g.classList.toggle("chain-sel", CHM.sel === g);
    if (!g._chainBound) {
      g._chainBound = true;
      g.addEventListener("pointerdown", () => { if (repairMode() === "chain") { CHM.sel = g; for (const x of blocks) x.classList.toggle("chain-sel", x === g); liveScriptRender(); } }, true);
    }
  }
  // "+" between blocks: a filter group placed right there, then the library to fill it
  for (let i = 0; i + 1 < blocks.length; i++) {
    const before = blocks[i], next = blocks[i + 1];
    if (before.dataset.pin === "source" && next.dataset.gid === "telecine") continue;
    const add = el("div", "chain-add"), b = el("button", null, "+");
    b.title = "Add a filter or an app stage here";
    b.onclick = () => chainAddAfter(before);
    add.appendChild(b);
    next.parentNode.insertBefore(add, next);
  }
}
function chainAnchor(g) {                              // the Guided group a new filter group goes after
  if (g.dataset.pin === "source") return "";
  if (g.classList.contains("vsf-group")) { const vg = S.vs_groups.find((x) => x.id === g.dataset.gid); return vg ? vsfAfter(vg) : "colour"; }
  return g.dataset.gid || "colour";
}
function chainAddAfter(g) {
  vsfMigrate();
  const ng = vsfNewGroup(chainAnchor(g));
  S.vs_groups.push(ng); vsgChanged();
  const target = S.vs_groups.find((x) => x.id === ng.id) || ng;
  vsfBrowser(target, null, { fresh: true, where: "→ the chain, after " + ((g.querySelector("h3") || {}).textContent || "this block") });
}
// ---- the live script beside the chain
const LIVESCR = { text: "" };
function liveScriptRender(text) {
  if (text != null) LIVESCR.text = text;
  const box = $("liveLines");
  if (!box || repairMode() !== "guided" || document.body.classList.contains("live-off")) return;
  const sel = CHM.sel, key = sel ? (sel.classList.contains("vsf-group") ? "vsf:" + sel.dataset.gid : sel.dataset.gid || sel.dataset.pin) : null;
  let re = key && CHAIN_LINES[key];
  if (key && key.startsWith("vsf:")) { const vg = S.vs_groups.find((x) => "vsf:" + x.id === key); re = vg ? new RegExp(`vs_filters\\(clip, job, "${vsfPoint(vg)}"\\)`) : null; }
  const lines = (LIVESCR.text || "").split("\n");
  let first = -1, html = "";
  lines.forEach((l, i) => {
    const hit = re && re.test(l) && /^\s*\w+\s*=\s*(dr|core|vs[a-z_]*)\.|^\s*clip\.set_output/.test(l);   // code lines, not the job settings
    if (hit && first < 0) first = i;
    html += `<div class="ln${hit ? " hl" : ""}" data-l="${i}"><span>${i + 1}</span><span>${pyHighlight(l) || " "}</span></div>`;
  });
  box.innerHTML = html;
  $("liveCount").textContent = `${lines.length} lines`;
  if (first >= 0) box.children[first].scrollIntoView({ block: "center" });
  for (const d of box.children) d.onclick = () => liveLineClick(lines[+d.dataset.l]);
}
function liveLineClick(line) {                         // a line -> its block (opened, scrolled to)
  const panel = document.querySelector('.insp [data-panel="adjust"]');
  let g = null;
  const m = /vs_filters\(clip, job, "(\w+)"\)/.exec(line);
  if (m) { const vg = S.vs_groups.find((x) => vsfPoint(x) === m[1]); if (vg) g = panel.querySelector(`.vsf-group[data-gid="${vg.id}"]`); }
  if (!g) for (const [gid, re] of Object.entries(CHAIN_LINES)) if (re.test(line)) { g = panel.querySelector(`:scope > .group[data-gid="${gid}"], :scope > .chain-pin[data-pin="${gid}"]`); if (g) break; }
  if (!g) return;
  CHM.sel = g; chainDecorate(); liveScriptRender();
  g.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (!g.classList.contains("open")) { const h = g.querySelector(".group-head"); if (h) h.click(); }
}

