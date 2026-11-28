// Script mode: the script editor (Repair › Script, also the Script dock tab). A plain textarea with a coloured copy
// behind it, line numbers, completion (Ctrl+Space, or after a "."), errors on their line, snippets, and
// "preview to the cursor" (Ctrl+Enter): the preview shows `clip` as it is after that line (S.scriptLine).
"use strict";

const SCE = { api: null, ac: null, acSel: 0, acItems: [], err: null, info: null, built: false, rightWas: null };
const PY_KW = new Set(("False None True and as assert async await break class continue def del elif else except finally for from global if " +
                       "import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
const SCRIPT_NAMES = ["clip", "job", "core", "vs", "dr", "ctx", "EPISODE", "vstools", "vsdenoise", "vsaa", "vsdehalo", "vsdeinterlace", "vsscale",
                      "vsmasktools", "vsrgtools", "vsdeband", "vskernels", "vsexprtools"];
const SNIPPETS = [
  ["Deband", "import vsdeband\nclip = vsdeband.f3k_deband(clip, radius=16, thr=24)                      # smooth banding in flat gradients"],
  ["Denoise (FluxSmooth)", "clip = core.zsmooth.FluxSmoothST(clip, temporal_threshold=7, spatial_threshold=7)   # grain / noise"],
  ["Mask merge", "import vstools, vsmasktools\nstrong = core.zsmooth.FluxSmoothST(clip, temporal_threshold=12, spatial_threshold=12)   # way 1: strong\nmask = vsmasktools.Kirsch.edgemask(vstools.get_y(clip)).std.Maximum()   # the outlines\nclip = core.std.MaskedMerge(strong, clip, mask, first_plane=True)  # strong in flat areas, lines kept"],
  ["Frame range", "alt = core.std.BoxBlur(clip, hradius=2, vradius=2)                     # your filter\nclip = dr.splice_src(clip, alt, [(1000, 1500)])                         # only source frames 1000-1499"],
  ["An app block", "clip = dr.block(clip, job, 'halos', {'method': 'ai', 'strength': 1.0}, dr.chain_context(job))   # any Repair stage as one line"],
  ["Your AI model", "clip = dr.block(clip, job, 'ai', {'call': 'ai.model', 'args': {'model': r'C:\\models\\model.onnx', 'strength': '1'}}, dr.chain_context(job))"],
  ["Look at a frame prop", "print(clip.get_frame(0).props)                                          # appears in the Log"],
];

function sced() { return { refresh: scedRefresh }; }
function scedBuild() {
  if (SCE.built) return;
  SCE.built = true;
  const box = $("scriptBox");
  const wrap = el("div", "sced"), gut = el("div", "sced-gut"), area = el("div", "sced-area"), hl = el("pre", "sced-hl");
  gut.id = "scedGut"; hl.id = "scedHl";
  box.parentNode.insertBefore(wrap, box);
  area.append(hl, box); wrap.append(gut, area);
  const ac = el("div", "sced-ac hidden"); ac.id = "scedAc"; wrap.appendChild(ac);
  const bar = el("div", "sced-bar");
  const pos = el("span", "mono muted", ""); pos.id = "scedPos";
  const line = el("span", "chip hidden"); line.id = "scedLine";
  const prev = el("button", null, "▶ Preview to cursor"); prev.title = "Show the picture as it is after the line the cursor is on (Ctrl+Enter)"; prev.onclick = () => scedPreviewLine();
  const snip = el("button", "ghost", "Snippets ▾"); snip.onclick = (e) => scedSnippets(e.currentTarget);
  const help = el("button", "ghost", "?"); help.title = "Ctrl+Space: complete (plugins, vs-jetpack, the app's dr.* functions) · Ctrl+Enter: preview to the cursor line · F5: check · Tab: indent";
  bar.append(prev, line, snip, help, el("span", "push"), pos);
  wrap.parentNode.insertBefore(bar, wrap);
  const ext = el("div", "note sced-ext hidden"); ext.id = "scedExt"; wrap.parentNode.insertBefore(ext, wrap);
  box.wrap = "off";
  box.addEventListener("input", () => { SCE.err = null; scedRefresh(); scedAcMaybe(); });
  box.addEventListener("scroll", () => { hl.scrollTop = box.scrollTop; hl.scrollLeft = box.scrollLeft; gut.scrollTop = box.scrollTop; scedAcHide(); });
  box.addEventListener("click", () => { scedPos(); scedAcHide(); });
  box.addEventListener("keyup", (e) => { if (!["ArrowUp", "ArrowDown"].includes(e.key) || !SCE.ac) scedPos(); });
  box.addEventListener("keydown", scedKey, true);
  box.addEventListener("blur", () => setTimeout(scedAcHide, 150));
  scedRefresh();
}
function scedRefresh() {
  const box = $("scriptBox"), hl = $("scedHl"), gut = $("scedGut");
  if (!hl) return;
  const lines = box.value.split("\n");
  hl.innerHTML = pyHighlight(box.value) + "\n";
  const cur = scedCursorLine();
  let g = "";
  for (let i = 1; i <= lines.length; i++) {
    const cls = SCE.err && SCE.err.line === i ? "err" : S.scriptLine === i ? "run" : i === cur ? "cur" : "";
    g += `<div class="${cls}">${S.scriptLine === i ? "▶ " : ""}${i}</div>`;
  }
  gut.innerHTML = g + "<div>&nbsp;</div>";
  hl.scrollTop = box.scrollTop; gut.scrollTop = box.scrollTop;
  const xn = $("scedExt");
  if (xn) {
    const ext = scriptIsExternal(S.script || box.value) && !!S.script;
    xn.classList.toggle("hidden", !ext);
    if (ext) xn.textContent = "Your own script: it runs as written (its own source and filters). Each episode is its part of the output, the sound and subtitles come from "
      + (S.info ? basename(S.info.path) : "the file opened here") + "; the export keeps the frame rate it makes."
      + ((SCE.ext || []).length ? ` It loads ${SCE.ext.map(basename).join(", ")} - a different file: the sound only matches if the picture does.` : "");
  }
  const lc = $("scedLine");
  lc.classList.toggle("hidden", !S.scriptLine);
  if (S.scriptLine) {
    lc.innerHTML = "";
    lc.append(el("span", null, `Preview: up to line ${S.scriptLine}${SCE.info ? " · " + SCE.info : ""}`));
    const x = el("button", "link", "✕"); x.title = "Preview the whole script again"; x.onclick = () => { S.scriptLine = null; SCE.info = null; scedRefresh(); schedulePreview(); };
    lc.appendChild(x);
  }
  scedPos();
}
function esc(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function pyHighlight(src) {
  const re = /("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|[rbfRBF]{0,2}"(?:\\.|[^"\\\n])*"?|[rbfRBF]{0,2}'(?:\\.|[^'\\\n])*'?)|(#[^\n]*)|(\b\d+(?:\.\d*)?(?:e[-+]?\d+)?\b)|([A-Za-z_]\w*)(\s*\()?|([=+\-*/%<>!&|^~@]+)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    if (m[1]) out += `<span class="s">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="c">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="n">${m[3]}</span>`;
    else if (m[4]) {
      const w = m[4];
      if (PY_KW.has(w)) out += `<span class="k">${w}</span>`;
      else if (m[5]) out += `<span class="f">${w}</span>` + esc(m[5]);
      else if (w === "clip" || w === "job" || w === "core" || w === "dr" || w === "vs") out += `<span class="v">${w}</span>`;
      else out += w;
      if (!m[5] && m[5] !== undefined) out += "";
    } else if (m[6]) out += `<span class="o">${esc(m[6])}</span>`;
    last = re.lastIndex;
  }
  return out + esc(src.slice(last));
}
function scedCursorLine() { const box = $("scriptBox"); return box.value.slice(0, box.selectionStart).split("\n").length; }
function scedPos() {
  const box = $("scriptBox"), p = $("scedPos");
  if (!p) return;
  const before = box.value.slice(0, box.selectionStart), ln = before.split("\n").length, col = before.length - before.lastIndexOf("\n");
  p.textContent = `line ${ln}, col ${col}`;
  for (const [i, d] of [...$("scedGut").children].entries()) d.classList.toggle("cur", i + 1 === ln && !d.classList.contains("err") && !d.classList.contains("run"));
}
function scedKey(e) {
  const box = $("scriptBox");
  if (SCE.ac) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); e.stopImmediatePropagation(); SCE.acSel = (SCE.acSel + (e.key === "ArrowDown" ? 1 : -1) + SCE.acItems.length) % SCE.acItems.length; scedAcDraw(); return; }
    if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); e.stopImmediatePropagation(); scedAcAccept(); return; }
    if (e.key === "Escape") { e.preventDefault(); scedAcHide(); return; }
  }
  if (e.key === " " && e.ctrlKey) { e.preventDefault(); scedAcShow(true); return; }
  if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); scedPreviewLine(); return; }
  if (e.key === "F5") { e.preventDefault(); checkScript(); return; }
  if (e.key === "Enter" && !e.ctrlKey && !e.shiftKey) {       // keep the indentation (and indent after a colon)
    e.preventDefault(); e.stopImmediatePropagation();
    const a = box.selectionStart, ls = box.value.lastIndexOf("\n", a - 1) + 1, line = box.value.slice(ls, a);
    let ind = (/^\s*/.exec(line) || [""])[0]; if (/:\s*$/.test(line)) ind += "    ";
    box.setRangeText("\n" + ind, a, box.selectionEnd, "end"); box.dispatchEvent(new Event("input"));
  }
}
// ---------------------------------------------------------------- completion
async function scedApi() {
  if (!SCE.api) {
    try { SCE.api = await rpc("script_api", {}); } catch { SCE.api = { dr: [], blocks: [] }; }
    try { await vsfLibrary(); } catch { /* plugins unknown */ }
  }
  return SCE.api;
}
function scedWordAt() {                                // the dotted name being typed: "core.std.Box"
  const box = $("scriptBox"), a = box.selectionStart, before = box.value.slice(0, a);
  const m = /([A-Za-z_][\w.]*)$/.exec(before);
  return m ? m[1] : "";
}
function scedCandidates(word) {
  const api = SCE.api || { dr: [] }, lib = VSF_LIB || [];
  const dot = word.lastIndexOf("."), base = dot >= 0 ? word.slice(0, dot) : "", part = (dot >= 0 ? word.slice(dot + 1) : word).toLowerCase();
  const out = [];
  const add = (label, insert, sig, doc, kind) => { if (label.toLowerCase().startsWith(part) || (part.length > 1 && label.toLowerCase().includes(part))) out.push({ label, insert, sig, doc, kind }); };
  if (base === "dr") for (const f of api.dr) add(f.name, f.name + "(", f.sig, f.doc, "app");
  else if (base === "core") {
    const ns = [...new Set(lib.filter((f) => f.kind === "vs").map((f) => f.call.split(".")[0]))].sort();
    for (const n of ns) add(n, n + ".", "", `${lib.filter((f) => f.kind === "vs" && f.call.startsWith(n + ".")).length} functions`, "plugin");
    add("std", "std.", "", "VapourSynth's own functions", "plugin");
  } else if (base.startsWith("core.")) {
    const ns = base.slice(5);
    for (const f of lib.filter((x) => x.kind === "vs" && x.call.startsWith(ns + ".")))
      add(f.call.split(".")[1], f.call.split(".")[1] + "(", "(" + (f.args || []).map((a) => a.name + (a.opt ? "?" : "")).join(", ") + ")", f.doc || f.note || f.name, "plugin");
  } else if (base) {
    for (const f of lib.filter((x) => x.kind === "jet" && x.call.startsWith(base + ".") && !x.call.slice(base.length + 1).includes(".")))
      add(f.call.slice(base.length + 1), f.call.slice(base.length + 1) + "(", "(" + (f.args || []).map((a) => a.name).join(", ") + ")", f.doc || f.name, "vs-jetpack");
    const subs = new Set(lib.filter((x) => x.kind === "jet" && x.call.startsWith(base + ".")).map((x) => x.call.slice(base.length + 1).split(".")[0]));
    for (const s of subs) if (!out.some((o) => o.label === s)) add(s, s + ".", "", "", "vs-jetpack");
  } else {
    const vars = new Set([...$("scriptBox").value.matchAll(/^\s*([A-Za-z_]\w*)\s*=/gm)].map((m) => m[1]));
    for (const n of [...SCRIPT_NAMES, ...vars]) add(n, n, "", "", "name");
    for (const k of PY_KW) if (k.length > 2) add(k, k, "", "", "keyword");
  }
  const seen = new Set();
  return out.filter((o) => !seen.has(o.label) && seen.add(o.label)).sort((a, b) => (a.label.toLowerCase().startsWith(part) ? 0 : 1) - (b.label.toLowerCase().startsWith(part) ? 0 : 1) || a.label.localeCompare(b.label)).slice(0, 60);
}
function scedAcMaybe() {
  const w = scedWordAt();
  if (w.endsWith(".") || (SCE.ac && w)) scedAcShow(false);
  else if (SCE.ac && !w) scedAcHide();
}
async function scedAcShow(force) {
  await scedApi();
  const w = scedWordAt();
  if (!w && !force) return scedAcHide();
  SCE.acItems = scedCandidates(w);
  if (!SCE.acItems.length) return scedAcHide();
  SCE.ac = w; SCE.acSel = 0;
  scedAcDraw();
}
function scedAcDraw() {
  const ac = $("scedAc"), box = $("scriptBox");
  ac.innerHTML = "";
  SCE.acItems.forEach((it, i) => {
    const r = el("div", "row" + (i === SCE.acSel ? " on" : ""));
    r.append(el("span", null, it.label), el("small", "mono", it.sig ? it.sig.slice(0, 60) : ""), el("i", null, it.kind));
    r.onmousedown = (e) => { e.preventDefault(); SCE.acSel = i; scedAcAccept(); };
    ac.appendChild(r);
  });
  const cur = SCE.acItems[SCE.acSel];
  if (cur && (cur.doc || cur.sig)) { const d = el("div", "doc"); d.append(el("code", null, cur.label + (cur.sig || "")), el("div", null, cur.doc || "")); ac.appendChild(d); }
  // place it under the cursor
  const before = box.value.slice(0, box.selectionStart), ln = before.split("\n").length - 1, col = before.length - before.lastIndexOf("\n") - 1;
  const lh = 20, cw = 7.5;
  ac.style.top = Math.max(4, 8 + (ln + 1) * lh - box.scrollTop) + "px";
  ac.style.left = Math.min(Math.max(50, 52 + col * cw - box.scrollLeft), box.clientWidth - 300) + "px";
  ac.classList.remove("hidden");
  const on = ac.querySelector(".row.on"); if (on) on.scrollIntoView({ block: "nearest" });
}
function scedAcHide() { SCE.ac = null; const ac = $("scedAc"); if (ac) ac.classList.add("hidden"); }
function scedAcAccept() {
  const it = SCE.acItems[SCE.acSel]; if (!it) return scedAcHide();
  const box = $("scriptBox"), a = box.selectionStart, w = scedWordAt(), dot = w.lastIndexOf(".");
  const start = a - (dot >= 0 ? w.length - dot - 1 : w.length);
  box.setRangeText(it.insert, start, a, "end");
  scedAcHide();
  box.dispatchEvent(new Event("input"));
  if (it.insert.endsWith(".")) setTimeout(() => scedAcShow(false), 0);
  else if (it.sig) status(`${it.label}${it.sig}${it.doc ? "  -  " + it.doc.slice(0, 160) : ""}`);
}
// ---------------------------------------------------------------- preview to a line, snippets, errors
async function scedPreviewLine() {
  if (!S.script) { S.script = $("scriptBox").value; scriptStateUi(); }
  const n = scedCursorLine();
  S.scriptLine = n; SCE.info = "…"; scedRefresh(); schedulePreview(30);
  try {
    const r = await rpc("check_script", { state: backendState([S.current]), episode: S.current, text: $("scriptBox").value, upto: n });
    if (S.scriptLine !== n) return;
    if (r.ok) { SCE.info = `${r.format} ${r.width}×${r.height} · ${r.fps} fps · ${r.frames} frames`; SCE.err = null; }
    else { SCE.info = "error"; SCE.err = { line: r.line, msg: r.error }; status(`Line ${r.line || "?"}: ${r.error}`, "err"); }
  } catch { SCE.info = null; }
  scedRefresh();
}
function scedSnippets(anchor) {
  document.querySelector(".ch-menu")?.remove();
  const menu = el("div", "ch-menu");
  menu.appendChild(el("div", "ch-mcat", "Insert at the cursor"));
  for (const [name, code] of SNIPPETS) {
    const r = el("div", "ch-mi"); r.append(el("b", null, name), el("small", "mono", code.split("\n")[0].slice(0, 70)));
    r.onclick = () => {
      const box = $("scriptBox"), a = box.selectionStart, ls = box.value.lastIndexOf("\n", a - 1) + 1;
      box.setRangeText(code + "\n", ls, ls, "end"); box.focus(); box.dispatchEvent(new Event("input")); menu.remove();
    };
    menu.appendChild(r);
  }
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  menu.style.top = (r.bottom + 4) + "px"; menu.style.left = Math.max(8, Math.min(window.innerWidth - 380, r.left)) + "px";
  const off = (e) => { if (!menu.contains(e.target) && e.target !== anchor) { menu.remove(); document.removeEventListener("pointerdown", off, true); } };
  setTimeout(() => document.addEventListener("pointerdown", off, true), 0);
}
function scriptIsExternal(t) { return !!t && !(t.includes("EPISODE = ") && t.includes("# ---- end of episode ----")); }
async function scriptOpen(given) {                  // given = {path, text} (already read), else the file dialog
  const r = given && given.text != null ? given : await window.api.openText([{ name: "VapourSynth script", extensions: ["vpy", "py"] }]);
  if (!r) return;
  if (!S.info) { status("Open the video the script is for first: its sound and subtitles come from there.", "err"); return; }
  if (S.script && S.script !== r.text && !confirm("Replace the script in the editor with " + basename(r.path) + "?")) return;
  S.script = r.text; S.scriptFile = basename(r.path); S.scriptLine = null;
  $("scriptBox").value = r.text;
  goto("adjust"); scriptStateUi(); rmodeSync(); scedRefresh(); schedulePreview(30); sync();
  log(`Script loaded: ${r.path}`, "ok");
  try {
    const c = await rpc("check_script", { state: backendState([S.current]), episode: S.current, text: r.text });
    if (!c.ok) { scedShowError(c.line, c.error); status(`${basename(r.path)}: line ${c.line || "?"}: ${c.error}`, "err"); return; }
    SCE.ext = c.other_sources || [];
    const w = SCE.ext.length ? ` Note: it loads ${SCE.ext.map(basename).join(", ")}, not ${basename(S.info.path)} - the sound and subtitles come from ${basename(S.info.path)}.` : "";
    status(`${basename(r.path)}: ${c.width}x${c.height} ${c.format}, ${c.fps} fps, ${c.frames} frames.${w}`, w ? "err" : "ok");
    scedRefresh();
  } catch { /* logged */ }
}
function scedShowError(line, msg) { SCE.err = line ? { line, msg } : null; scedRefresh(); }
function scedPreviewError(e) {                       // a script error in the preview: mark its line
  const m = /edited script, line (\d+)|File "edited script", line (\d+)/.exec(String(e && e.message || e));
  if (m) scedShowError(+(m[1] || m[2]), String(e.message || e));
}
function scedFirstClip() {                           // the cursor on the first `clip =` line (the job dict above is long)
  const box = $("scriptBox"), i = box.value.search(/^clip = /m);
  if (i < 0) return;
  box.setSelectionRange(i, i);
  box.scrollTop = Math.max(0, (box.value.slice(0, i).split("\n").length - 4) * 20);
  scedRefresh();
}
// Script mode puts the editor in the Inspector (made wide); otherwise it lives in the Script dock tab
// The script editor lives in a window (Repair's </> Script, View > Script window, Ctrl+Shift+S): the script of your settings
// (it follows them), or yours - type in it, Open .vpy… loads a script written elsewhere, Save .vpy… keeps it.
function scriptHome() {
  scedBuild();
  const grp = document.querySelector(".script-group"), target = $("scriptWinBody");
  if (!grp || !target) return;
  if (grp.parentNode !== target) target.insertBefore(grp, target.firstChild);
  scedRefresh();
}
function scriptWinOpen() { const w = $("scriptWin"); return !!w && !w.classList.contains("hidden"); }
function scriptWinShow(on) {
  const w = $("scriptWin");
  if (!w) return;
  scriptHome();
  w.classList.toggle("hidden", !on);
  document.body.classList.toggle("script-open", on);
  const b = $("btnScriptSide"); if (b) b.classList.toggle("on", on);
  if (on) {
    if (!S.scriptLine) SCE.info = null;
    $("swEp").textContent = String((S.current || 0) + 1);
    if (S.info) refreshScript(!S.script);                // the script of the settings as they are now (yours stays as typed)
    scedRefresh();
    setTimeout(() => { const t = $("scriptBox"); if (t && !S.script) t.scrollTop = 0; }, 50);
  } else if (S.scriptLine) { S.scriptLine = null; schedulePreview(); }
  setTimeout(layoutImage, 0);
}
function scriptWinReveal(g) {                            // a panel was clicked: its lines in the script (if the window is open)
  if (!scriptWinOpen() || !g || typeof CHAIN_LINES === "undefined") return;
  const key = g.classList.contains("vsf-group") ? null : g.dataset.gid || g.dataset.pin;
  let re = key && CHAIN_LINES[key];
  if (g.classList.contains("vsf-group")) { const vg = S.vs_groups.find((x) => x.id === g.dataset.gid); re = vg ? new RegExp(`vs_filters\\(clip, job, "${vsfPoint(vg)}"\\)`) : null; }
  if (!re) return;
  const box = $("scriptBox"), lines = box.value.split("\n");
  const i = lines.findIndex((l) => re.test(l) && /^\s*\w+\s*=\s*(dr|core|vs[a-z_]*)\./.test(l));
  if (i >= 0) { box.scrollTop = Math.max(0, i * 20 - 80); box.dispatchEvent(new Event("scroll")); }
}
function initScriptWindow() {
  $("swClose").onclick = () => scriptWinShow(false);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && scriptWinOpen() && !document.querySelector(".dialog-back:not(.hidden)")) scriptWinShow(false);
    if (e.ctrlKey && e.shiftKey && (e.key === "S" || e.key === "s")) { e.preventDefault(); scriptWinShow(!scriptWinOpen()); }
  });
  const head = $("swHead"), w = $("scriptWin");                // drag the window by its title
  head.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button")) return;
    const r = w.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
    head.setPointerCapture(e.pointerId);
    const move = (ev) => { w.style.left = Math.max(0, Math.min(innerWidth - 120, ev.clientX - dx)) + "px"; w.style.top = Math.max(0, Math.min(innerHeight - 60, ev.clientY - dy)) + "px"; w.style.right = "auto"; w.style.bottom = "auto"; w.style.height = r.height + "px"; };
    head.addEventListener("pointermove", move);
    head.addEventListener("pointerup", () => head.removeEventListener("pointermove", move), { once: true });
  });
}
document.addEventListener("DOMContentLoaded", () => { try { scedBuild(); initRepairModes(); initScriptWindow(); } catch (e) { console.error(e); } });
