// Timeline (bottom dock), laid out like an editor's: tools | track heads | tracks under a timecode ruler.
//   L1..Ln   adjustment layers (Colour), then a lane per Repair step / filter group that has frame ranges
//   V1       the episode as a filmstrip, cut into parts with the Razor; parts cut out are hatched and left out of
//            the export (server.run_edited exports the kept parts and joins them)
//   (sound and subtitles are not shown; a cut still cuts them in the export)
// Positions are source frames of the whole title (S.episodes / S.layers / S.ranges use the same numbers); the
// episode is the clip on V1, its ends trim the episode. S.splits = edit points, S.cuts = [[a, b)] parts cut out.
"use strict";

const TLX = { v0: null, v1: null, fit: null, tool: "sel", snap: true, drag: null, sel: null, focus: null, seekF: null,
              playSrc: null, media: {}, thumbs: new Map(), thumbBusy: false, thumbQ: new Set(), raf: 0, hover: null };
const TLX_TOOLS = { sel: ["Selection", "V"], razor: ["Razor", "C"], ripple: ["Ripple trim", "R"], hand: ["Hand", "H"], zoom: ["Zoom", "Z"] };
const TLX_HINT = {
  sel: "Click a part to select it · drag its edges to trim (between two parts: moves the edit) · Del cuts it out / brings it back · Ctrl+K adds an edit at the playhead",
  razor: "Click to cut the video, sound and subtitles at that frame (on an adjustment layer: cuts the layer) · Shift = no snapping",
  ripple: "Drag a part's edge: the part gets shorter, what you trim off is cut out (the export closes the gap) · Q / W cut to the playhead",
  hand: "Drag to scroll the timeline",
  zoom: "Click to zoom in there · Alt+click zooms out · \\ fits the episode",
};
const TH = { ruler: 22, l: 24, v: 54, a: 46, s: 20 };

// ---------------------------------------------------------------- helpers
function tcf(f) {                                   // hh:mm:ss:ff (frames at the nominal rate, like an editor)
  const s = Math.max(0, frameTime(f) + 0.002), nom = Math.round(fps()), sec = Math.floor(s + 1e-6);
  const ff = Math.min(nom - 1, Math.floor((s - sec) * nom + 1e-4));
  return [Math.floor(sec / 3600), Math.floor(sec / 60) % 60, sec % 60, ff].map((v) => String(v).padStart(2, "0")).join(":");
}
function tcs(sec) {                                 // seconds -> hh:mm:ss:ff
  const nom = Math.round(fps()), t = Math.max(0, sec) + 1e-6, s0 = Math.floor(t), ff = Math.min(nom - 1, Math.floor((t - s0) * nom));
  return [Math.floor(s0 / 3600), Math.floor(s0 / 60) % 60, s0 % 60, ff].map((v) => String(v).padStart(2, "0")).join(":");
}
function timeFrameF(s) {                            // time -> frame, fractional (ruler ticks sit exactly on their time)
  if (S.info && S.info.frame_ms) return timeFrame(s);
  let c = s;
  for (const [tj, d] of (S.info && S.info.tl_jumps) || []) if (s > tj) c -= Math.min(d, s - tj);
  return c * fps();
}
function tlxTotal() { return S.info ? S.info.frames : 1; }
function tlxFit() {
  const [a, b] = epRange(), m = Math.max(fps(), (b - a) * 0.025);
  TLX.v0 = Math.max(0, a - m); TLX.v1 = Math.min(tlxTotal(), b + m);
  if (TLX.v1 - TLX.v0 < 48) TLX.v1 = TLX.v0 + 48;
}
function ltWindow() {
  const key = `${S.info ? S.info.path : ""}#${S.current}#${epRange().join("-")}`;
  if (TLX.v0 == null || (TLX.fit !== key && !TLX.drag)) { TLX.fit = key; tlxFit(); }
  return [TLX.v0, TLX.v1];
}
function ltX(f, w) { return (100 * (f - w[0]) / Math.max(1e-9, w[1] - w[0])); }
function ltFrame(x) {
  const r = $("ltBody").getBoundingClientRect(), w = ltWindow();
  return Math.round(w[0] + Math.max(0, Math.min(1, (x - r.left) / Math.max(1, r.width))) * (w[1] - w[0]));
}
function ltFrameRaw(x) { const r = $("ltBody").getBoundingClientRect(), w = ltWindow(); return w[0] + ((x - r.left) / Math.max(1, r.width)) * (w[1] - w[0]); }
function ltTip(text, x) {
  const tip = $("ltTip"), r = $("ltBody").getBoundingClientRect();
  tip.textContent = text; tip.style.left = Math.max(40, Math.min(r.width - 40, x - r.left)) + "px"; tip.style.top = "2px";                    // in the ruler strip: never over the layers
  tip.classList.remove("hidden");
}
function tlxZoom(factor, at) {                       // at = frame that stays under the mouse
  const [v0, v1] = ltWindow(), N = tlxTotal();
  const span = Math.max(Math.max(24, 2 * fps()), Math.min(N * 1.02, (v1 - v0) * factor));
  const p = at == null ? 0.5 : (at - v0) / Math.max(1, v1 - v0);
  const c = at == null ? (v0 + v1) / 2 : at;
  TLX.v0 = c - p * span; TLX.v1 = TLX.v0 + span;
  tlxClamp(); tlxRender();
}
function tlxClamp() {
  const N = tlxTotal(), span = TLX.v1 - TLX.v0;
  if (TLX.v0 < 0) { TLX.v0 = 0; TLX.v1 = span; }
  if (TLX.v1 > N) { TLX.v1 = N; TLX.v0 = Math.max(0, N - span); }
}
function tlxHead() { return TLX.seekF ?? (PLAY.on && TLX.playSrc != null ? TLX.playSrc : null) ?? S.src ?? headSrcFrame(); }

// ---------------------------------------------------------------- the edit: parts, edit points, cut-out parts
function inCut(f) { return (S.cuts || []).some(([a, b]) => f >= a && f < b); }
function epSegs() {
  const [a, b] = epRange();
  const pts = new Set([a, b]);
  for (const f of S.splits || []) if (f > a && f < b) pts.add(f);
  for (const [c0, c1] of S.cuts || []) { if (c0 > a && c0 < b) pts.add(c0); if (c1 > a && c1 < b) pts.add(c1); }
  const bd = [...pts].sort((x, y) => x - y), out = [];
  for (let i = 0; i + 1 < bd.length; i++) out.push({ a: bd[i], b: bd[i + 1], cut: inCut((bd[i] + bd[i + 1]) / 2) });
  return out;
}
function setSegs(segs) {                             // write an edited list of parts back into S.splits / S.cuts
  const [a, b] = epRange();
  S.splits = (S.splits || []).filter((f) => f <= a || f >= b);
  const keep = [];
  for (const [c0, c1] of S.cuts || []) {             // other episodes' cuts (and the parts of ours outside it) stay
    if (c1 <= a || c0 >= b) keep.push([c0, c1]);
    else { if (c0 < a) keep.push([c0, a]); if (c1 > b) keep.push([b, c1]); }
  }
  const ss = segs.filter((s) => s.b > s.a);
  for (let i = 1; i < ss.length; i++) S.splits.push(ss[i].a);
  for (const s of ss) if (s.cut) keep.push([s.a, s.b]);
  keep.sort((x, y) => x[0] - y[0]);
  const merged = [];
  for (const c of keep) { const m = merged[merged.length - 1]; if (m && c[0] <= m[1]) m[1] = Math.max(m[1], c[1]); else merged.push([...c]); }
  S.cuts = merged;
  S.splits = [...new Set(S.splits)].sort((x, y) => x - y);
}
function editsChanged(msg) {
  tlxRender(); histArm(); sync();
  if (msg) status(msg);
}
function cutFrames() { const [a, b] = epRange(); return (S.cuts || []).reduce((n, [c0, c1]) => n + Math.max(0, Math.min(b, c1) - Math.max(a, c0)), 0); }
function addEdit(f, quiet) {
  const segs = epSegs(), i = segs.findIndex((s) => f > s.a && f < s.b);
  if (i < 0) return false;
  segs.splice(i, 1, { ...segs[i], b: f }, { ...segs[i], a: f });
  setSegs(segs);
  if (!quiet) editsChanged(`Edit at ${tcf(f)}: select a part and press Del to cut it out.`);
  return true;
}
function toggleCut(seg) {
  const segs = epSegs(), s = segs.find((x) => x.a === seg.a && x.b === seg.b);
  if (!s) return;
  s.cut = !s.cut; setSegs(segs);
  TLX.sel = { a: s.a, b: s.b };
  editsChanged(s.cut ? `Cut out ${tcf(s.a)} – ${tcf(s.b)} (${hmsms(frameTime(s.b) - frameTime(s.a)).replace(/^0:/, "")}): left out of the export. Del again brings it back.`
                     : `${tcf(s.a)} – ${tcf(s.b)} is back in.`);
}
function rippleTo(side) {                            // Q / W: cut from the previous edit to the playhead / playhead to the next
  const f = Math.round(tlxHead()), segs = epSegs(), i = segs.findIndex((s) => f > s.a && f < s.b);
  if (i < 0) return;
  const s = segs[i];
  if (side === "q") segs.splice(i, 1, { a: s.a, b: f, cut: true }, { ...s, a: f });
  else segs.splice(i, 1, { ...s, b: f }, { a: f, b: s.b, cut: true });
  setSegs(segs);
  editsChanged(side === "q" ? `Cut out from the edit before to the playhead (${tcf(s.a)} – ${tcf(f)}).` : `Cut out from the playhead to the next edit (${tcf(f)} – ${tcf(s.b)}).`);
}
function joinEdit(f) {
  const segs = epSegs(), i = segs.findIndex((s) => s.a === f);
  if (i <= 0 || segs[i].cut !== segs[i - 1].cut) return;
  segs.splice(i - 1, 2, { a: segs[i - 1].a, b: segs[i].b, cut: segs[i].cut });
  setSegs(segs); editsChanged(`Edit at ${tcf(f)} removed.`);
}
function editsReset() {
  const [a, b] = epRange();
  setSegs([{ a, b, cut: false }]); TLX.sel = null; editsChanged("All edits of this episode removed.");
}
function tlxSkipCut(src) {                           // Play: jump over parts cut out, like the export will
  const [a, b] = epRange();
  const c = (S.cuts || []).find(([c0, c1]) => src >= c0 && src < c1 && c1 > a && c0 < b);
  if (!c) return false;
  togglePlay(false);
  if (c[1] >= b) return true;                        // cut to the end: stop there
  rpc("locate", { state: backendState([S.current]), episode: S.current, src: c[1] })
    .then((r) => { S.frame = r.frame; togglePlay(true); }).catch(() => {});
  return true;
}

// ---------------------------------------------------------------- the filmstrip
function tlxThumbs(need) {
  const p = S.info.path;
  for (const f of need) if (!TLX.thumbs.has(p + "|" + f)) TLX.thumbQ.add(f);
  if (TLX.thumbBusy || !TLX.thumbQ.size) return;
  const frames = [...TLX.thumbQ].slice(0, 24);
  for (const f of frames) TLX.thumbQ.delete(f);
  TLX.thumbBusy = true;
  rpc("thumbs", { state: backendState([S.current]), episode: S.current, frames, h: 40 })
    .then((r) => { for (const [f, b] of Object.entries(r.thumbs || {})) TLX.thumbs.set(p + "|" + f, "data:image/jpeg;base64," + b); })
    .catch(() => { for (const f of frames) TLX.thumbs.set(p + "|" + f, ""); })
    .finally(() => {
      TLX.thumbBusy = false;
      if (TLX.thumbs.size > 1500) for (const k of [...TLX.thumbs.keys()].slice(0, 500)) TLX.thumbs.delete(k);
      tlxRender();
    });
}

// ---------------------------------------------------------------- drawing
function renderTimeline() { tlxRender(); }
function renderLayerTrack() { tlxRender(); }
function tlxRender() {
  if (TLX.raf) return;
  TLX.raf = requestAnimationFrame(() => { TLX.raf = 0; try { tlxDraw(); } catch (e) { console.error(e); } });
}
function tlxDraw() {
  const box = $("tlx");
  if (!box) return;
  const show = !!S.info && S.episodes.length > 0 && !(S.info && S.info.image);
  box.classList.toggle("empty", !show);
  if (!show) { $("ltLanes").innerHTML = ""; $("ltHeads").innerHTML = ""; $("ltRuler").innerHTML = ""; return; }
  const maxT = Math.max(...S.layers.map((l) => l.track || 1), 1);
  if (S.tracks < maxT) S.tracks = maxT;
  const w = ltWindow(), [a, b] = epRange(), body = $("ltBody"), bw = Math.max(1, body.clientWidth), ppf = bw / (w[1] - w[0]);
  const X = (f) => ltX(f, w) + "%";
  // sequence tabs (one per episode)
  const seq = $("tlxSeq");
  if (seq.dataset.n !== String(S.episodes.length) || seq.dataset.cur !== String(S.current)) {
    seq.dataset.n = S.episodes.length; seq.dataset.cur = S.current;
    seq.innerHTML = "";
    S.episodes.forEach((ep, k) => {
      const t = el("button", k === S.current ? "on" : "", `Episode ${k + 1}`);
      t.title = `${tc(ep[0])} – ${tc(ep[1])}`; t.onclick = () => selectEpisode(k); seq.appendChild(t);
    });
  }
  // timecode + edited length
  const head = tlxHead();
  $("tlxTc").textContent = TLX.frames ? String(Math.round(head)) : tcf(Math.round(head));
  $("tlxUnits").textContent = TLX.frames ? "FR" : "TC";
  const cf = cutFrames();
  $("tlxEdited").textContent = cf ? `edited ${hmsms(span(a, b) - (frameTime(a + cf) - frameTime(a))).replace(/\.\d+$/, "")} · ${(S.cuts || []).filter(([c0, c1]) => c1 > a && c0 < b).length} cut out` : "";
  $("tlxEdited").classList.toggle("hidden", !cf);
  for (const t of $$("#tlxTools button")) t.classList.toggle("on", t.dataset.tool === TLX.tool);
  $("tlxSnap").classList.toggle("on", TLX.snap);
  $("tlxHint").textContent = TLX_HINT[TLX.tool];
  body.dataset.tool = TLX.tool;
  // ruler
  const ruler = $("ltRuler"); ruler.innerHTML = "";
  const t0 = frameTime(w[0]), t1 = frameTime(w[1]), secPx = bw / Math.max(1e-6, t1 - t0);
  const STEPS = [1 / fps(), 2 / fps(), 5 / fps(), 10 / fps(), 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = STEPS.find((s) => s * secPx >= 90) || 3600, minor = step >= 1 ? step / 5 : step / 2;
  const band = el("div", "ep"); band.style.left = X(a); band.style.width = (ltX(b, w) - ltX(a, w)) + "%"; ruler.appendChild(band);
  if (TLX.frames) {                                  // frame numbers: ticks on round frame numbers
    const ppf2 = bw / Math.max(1, w[1] - w[0]);
    const FST = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000];
    const fs = FST.find((s) => s * ppf2 >= 90) || 100000, fm = fs >= 10 ? fs / 5 : fs;
    for (let f = Math.ceil(w[0] / fm) * fm; f < w[1]; f += fm) {
      const big = f % fs === 0, tk = el("i", big ? "big" : ""); tk.style.left = X(f); ruler.appendChild(tk);
      if (big) { const lab = el("span", "lab", String(f)); lab.style.left = X(f); ruler.appendChild(lab); }
    }
  } else for (let ts = Math.ceil(t0 / minor) * minor; ts < t1; ts += minor) {
    const f = timeFrameF(ts), big = Math.abs(ts / step - Math.round(ts / step)) < 1e-6;
    const tk = el("i", big ? "big" : ""); tk.style.left = X(f); ruler.appendChild(tk);
    if (big) { const lab = el("span", "lab", tcs(ts)); lab.style.left = X(f); ruler.appendChild(lab); }
  }
  for (const f of chapFrames()) if (f > w[0] && f < w[1]) { const c = el("b", "chap"); c.style.left = X(f); c.title = "Chapter " + tc(f); ruler.appendChild(c); }
  for (const bm of S.bookmarks || []) {
    if (bm.src < w[0] || bm.src > w[1]) continue;
    const m = el("div", "mk"); m.style.left = X(bm.src); m.title = `${bm.note} - click: go there · Alt+click: remove`;
    m.addEventListener("pointerdown", (e) => e.stopPropagation());
    m.onclick = (e) => { e.stopPropagation(); if (e.altKey) { S.bookmarks = S.bookmarks.filter((x) => x !== bm); tlxRender(); } else gotoSrc(bm.src); };
    ruler.appendChild(m);
  }
  // heads + lanes
  const heads = $("ltHeads"); heads.innerHTML = "";
  heads.appendChild(el("div", "sp"));
  const lanes = $("ltLanes"); lanes.innerHTML = "";
  const outside = (lane) => {                         // the title outside the episode: darker
    if (a > w[0]) { const o = el("div", "out"); o.style.left = "0"; o.style.width = X(a); lane.appendChild(o); }
    if (b < w[1]) { const o = el("div", "out"); o.style.left = X(b); o.style.right = "0"; lane.appendChild(o); }
  };
  // one set of layers for the colour adjustment layers and the filter ranges (Layer 1 is the one next to V1; higher = on top)
  if (!TLX.drag) { convertBaseLuts(); settleRows(); }
  const rItems = activeRangeItems(), eLayers = epLayers();
  const rows = Math.max(S.tracks, 1, ...rItems.map((x) => x.r.layer || 1), ...S.layers.map((l) => l.track || 1));
  for (let t = rows; t >= 1; t--) {
    const off = !!S.trackOff[t], name = rowLabel(t);
    const th = el("div", "th rg" + (off ? " off" : ""));
    th.title = `${name} - drag a bar up or down to put it on another layer`;
    const eye = el("button", "i", off ? "◌" : "◉"); eye.title = off ? "Show this layer's colour adjustments (they are used again)" : "Hide this layer's colour adjustments (they are not used)";
    eye.onclick = () => { if (off) delete S.trackOff[t]; else S.trackOff[t] = true; layersChanged(); };
    const xt = el("button", "i", "✕"); xt.title = `Delete ${name} and what is on it`; xt.onclick = (e) => { e.stopPropagation(); removeTrack(t); };
    const tb = el("span", "th-btns"); tb.append(eye, xt);
    th.append(el("span", null, name), tb); heads.appendChild(th);
    const lane = el("div", "lt-lane l rg-lane" + (off ? " off" : "")); lane.dataset.track = t; lane.dataset.layer = t;
    outside(lane);
    for (const { l, i } of eLayers) {
      if ((l.track || 1) !== t) continue;
      const x0 = Math.max(-1, ltX(l.start, w)), x1 = Math.min(101, ltX(l.end, w));
      if (x1 <= 0 || x0 >= 100) continue;
      const bar = el("div", "lt-layer" + (i === S.layerSel ? " sel" : "") + (l.on === false ? " off" : ""), l.name);
      bar.style.left = x0 + "%"; bar.style.width = Math.max(0.3, x1 - x0) + "%"; bar.style.background = layerColour(l);
      bar.dataset.i = i; bar.title = `${l.name} · ${name}: ${tc(l.start)} – ${tc(l.end)} · opacity ${Math.round(100 * (l.opacity ?? 1))} %`;
      lane.appendChild(bar);
    }
    for (const { key, r } of rItems) if ((r.layer || 1) === t) { const bar = rangeBar(key, r, w); if (bar) lane.appendChild(bar); }
    lanes.appendChild(lane);
  }
  const segs = epSegs();
  const segBoxes = (lane, cls) => {
    for (const s of segs) {
      const x0 = ltX(s.a, w), x1 = ltX(s.b, w);
      if (x1 <= 0 || x0 >= 100) continue;
      const sel = TLX.sel && TLX.sel.a === s.a && TLX.sel.b === s.b;
      const d = el("div", `part ${cls}` + (s.cut ? " cut" : "") + (sel ? " sel" : ""));
      d.style.left = Math.max(-1, x0) + "%"; d.style.width = (Math.min(101, x1) - Math.max(-1, x0)) + "%";
      d.dataset.a = s.a; d.dataset.b = s.b;
      lane.appendChild(d);
    }
  };
  // V1
  const thv = el("div", "th v"); thv.append(el("b", null, "V1"), el("span", null, "Video")); heads.appendChild(thv);
  const v = el("div", "lt-lane v media"); outside(v);
  const stripL = Math.max(0, ltX(a, w)), stripR = Math.min(100, ltX(b, w));
  if (stripR > stripL) {
    const strip = el("div", "strip"); strip.style.left = stripL + "%"; strip.style.width = (stripR - stripL) + "%";
    const dar = S.info.width && S.info.height ? (S.info.dar || 4 / 3) : 4 / 3, tw = Math.round(40 * dar);
    let q = Math.max(1, tw / ppf); q = Math.pow(2, Math.ceil(Math.log2(q)));   // frames per picture, a power of 2 (cache reuse)
    const need = [], f0 = Math.max(a, Math.floor(w[0] / q) * q);
    for (let f = f0; f < Math.min(b, w[1]); f += q) {
      const fr = Math.round(Math.min(b - 1, Math.max(a, f)));
      need.push(fr);
      const im = el("i"); const src = TLX.thumbs.get(S.info.path + "|" + fr);
      if (src) im.style.backgroundImage = `url(${src})`;
      const xa = Math.max(a, f), xb = Math.min(b, f + q);
      im.style.left = ((ltX(xa, w) - stripL) / (stripR - stripL) * 100) + "%";
      im.style.width = ((ltX(xb, w) - ltX(xa, w)) / (stripR - stripL) * 100) + "%";
      strip.appendChild(im);
    }
    v.appendChild(strip);
    if (need.length <= 80) tlxThumbs(need);
  }
  segBoxes(v, "vp");
  for (const s of segs) {                            // part names
    const x0 = Math.max(0, ltX(s.a, w)), x1 = Math.min(100, ltX(s.b, w));
    if (x1 - x0 < 3) continue;
    const lab = el("span", "plab", s.cut ? "cut out · not exported" : `${basename(S.info.path)} [V]`);
    lab.style.left = `calc(${x0}% + 5px)`; v.appendChild(lab);
  }
  lanes.appendChild(v);
  // edit points
  for (let i = 1; i < segs.length; i++) {
    const f = segs[i].a; if (f <= w[0] || f >= w[1]) continue;
    const e = el("div", "editpt"); e.style.left = X(f); e.title = `Edit at ${tcf(f)} · Alt+click: remove it`; lanes.appendChild(e);
  }
  // playhead
  $("ltHead").style.left = X(head);
  $("ltHead").classList.toggle("hidden", head < w[0] || head > w[1]);
  // zoom / scroll bar
  const N = tlxTotal(), th = $("tlxThumb");
  th.style.left = (100 * w[0] / N) + "%"; th.style.width = Math.max(1.5, 100 * (w[1] - w[0]) / N) + "%";
  const spanS = t1 - t0;
  $("tlxZoomLab").textContent = spanS >= 120 ? `${Math.round(spanS / 60)} min` : spanS >= 2 ? `${Math.round(spanS)} s` : `${Math.round(spanS * fps())} frames`;
}

// ---------------------------------------------------------------- interaction
function tlxSnapF(f, shift, skip) {
  if (shift || !TLX.snap) return f;
  const w = ltWindow(), px = $("ltBody").getBoundingClientRect().width, lim = 8 / Math.max(1, px) * (w[1] - w[0]);
  const cands = [...epRange(), Math.round(tlxHead()), ...chapFrames(), ...(S.bookmarks || []).map((m) => m.src)];
  for (const s of epSegs()) cands.push(s.a, s.b);
  for (const l of S.layers) if (l !== skip && !l.pending) cands.push(l.start, l.end);
  let best = f, d0 = lim;
  for (const c of cands) if (Math.abs(c - f) < d0) { d0 = Math.abs(c - f); best = c; }
  return best;
}
const ltSnap = tlxSnapF;
function tlxDropLine() {                            // the blue line that shows where a new track / layer will be made
  return $("rgDropLine") || (() => { const l = el("div", "rg-dropline hidden"); l.id = "rgDropLine"; $("ltBody").appendChild(l); return l; })();
}
function ltTrackAt(y) {
  for (const lane of $$("#ltLanes .lt-lane[data-track]")) { const r = lane.getBoundingClientRect(); if (y >= r.top && y < r.bottom) return +lane.dataset.track; }
  return null;
}
async function ltSeekExact(f) {
  try { const r = await rpc("locate", { state: backendState([S.current]), episode: S.current, src: f }); S.frame = r.frame; schedulePreview(); } catch { /* logged */ }
}
function tlxSetTool(t) { TLX.tool = t; $("tlxRazor").classList.add("hidden"); tlxRender(); }
function splitLayerAt(i, f) {
  const l = S.layers[i];
  if (!l || f <= l.start || f >= l.end) return;
  const r = structuredClone(l); r.start = f; l.end = f; r.name = l.name.replace(/( \(\d+\))?$/, "") + " (2)";
  S.layers.splice(i + 1, 0, r); S.layerSel = i + 1; layersChanged();
}
function trimEpisodeEnd(which, f) {                   // V1's first / last edge = the episode's In / Out
  const ep = S.episodes[S.current];
  if (which === 0) ep[0] = Math.max(0, Math.min(ep[1] - 24, f));
  else ep[1] = Math.min(tlxTotal(), Math.max(ep[0] + 24, f));
  trimFields(ep[0], ep[1], true);
}
function tlxUnits(frames) {                          // the transport and ruler in timecode or in frame numbers (remembered)
  TLX.frames = frames;
  try { localStorage.setItem("tcUnits", frames ? "frames" : "tc"); } catch { /* no storage */ }
  $("tlxTc").title = frames ? "Frame number of the playhead (source frames) - click for timecode" : "Timecode of the playhead (hours:minutes:seconds:frames) - click for frame numbers";
  if (S.info && S.episodes.length) { const [a, b] = epRange(); trimFields(a, b); }
  tlxRender();
}
function initTimeline() {
  try { TLX.frames = localStorage.getItem("tcUnits") === "frames"; } catch { TLX.frames = false; }
  $("tlxTc").onclick = () => tlxUnits(!TLX.frames);
  $("tlxUnits").onclick = () => tlxUnits(!TLX.frames);
  const body = $("ltBody");
  for (const [k, [name, key]] of Object.entries(TLX_TOOLS)) {
    const b = $("tlxTools").querySelector(`[data-tool="${k}"]`);
    if (b) { b.title = `${name} (${key})`; b.onclick = () => tlxSetTool(k); }
  }
  $("tlxSnap").onclick = () => { TLX.snap = !TLX.snap; tlxRender(); };
  $("ltAdd").onclick = () => addLayer();
  $("ltTrack").onclick = () => { S.tracks++; tlxRender(); };
  $("tlxMarker").onclick = () => bookmarkAdd();
  $("tlxReset").onclick = () => { if (epSegs().length > 1) editsReset(); };
  $("tlxZin").onclick = () => tlxZoom(0.5, tlxHead());
  $("tlxZout").onclick = () => tlxZoom(2, tlxHead());
  $("tlxFitBtn").onclick = () => { tlxFit(); tlxRender(); };
  const jump = (f) => { togglePlay(false); gotoSrc(f); };
  $("tlxStart").onclick = () => jump(epRange()[0]);
  $("tlxEnd").onclick = () => jump(epRange()[1] - 1);
  $("tlxPrevEdit").onclick = () => tlxEditJump(-1);
  $("tlxNextEdit").onclick = () => tlxEditJump(1);
  $("tlxBack").onclick = () => stepFrame(-1);
  $("tlxFwd").onclick = () => stepFrame(1);
  // the zoom / scroll bar: drag the middle to scroll, the ends to zoom
  $("tlxScroll").addEventListener("pointerdown", (e) => {
    if (!S.info) return;
    const r = $("tlxScroll").getBoundingClientRect(), N = tlxTotal(), w = ltWindow(), tr = $("tlxThumb").getBoundingClientRect();
    const mode = Math.abs(e.clientX - tr.left) < 8 ? "l" : Math.abs(e.clientX - tr.right) < 8 ? "r" : e.clientX > tr.left && e.clientX < tr.right ? "m" : "jump";
    if (mode === "jump") { const c = (e.clientX - r.left) / r.width * N, sp = w[1] - w[0]; TLX.v0 = c - sp / 2; TLX.v1 = c + sp / 2; tlxClamp(); tlxRender(); return; }
    const x0 = e.clientX, a0 = w[0], b0 = w[1];
    $("tlxScroll").setPointerCapture(e.pointerId);
    const mv = (ev) => {
      const d = (ev.clientX - x0) / r.width * N, minS = Math.max(24, 2 * fps());
      if (mode === "m") { TLX.v0 = a0 + d; TLX.v1 = b0 + d; }
      else if (mode === "l") TLX.v0 = Math.min(b0 - minS, a0 + d);
      else TLX.v1 = Math.max(a0 + minS, b0 + d);
      tlxClamp(); tlxRender();
    };
    const up = () => { $("tlxScroll").removeEventListener("pointermove", mv); };
    $("tlxScroll").addEventListener("pointermove", mv); $("tlxScroll").addEventListener("pointerup", up, { once: true });
  });
  body.addEventListener("pointerdown", (e) => {
    if (!S.info || e.button === 2 || S.busy) return;
    togglePlay(false);
    const fr = ltFrameRaw(e.clientX), f = Math.round(fr), w = ltWindow(), [a, b] = epRange();
    const onRuler = !!e.target.closest("#ltRuler");
    const bar = e.target.closest(".lt-layer:not(.rg-blk)"), part = e.target.closest(".part");
    const capture = () => { try { body.setPointerCapture(e.pointerId); } catch { /* synthetic */ } };
    if (onRuler) {                                   // scrub
      TLX.drag = { kind: "seek" }; TLX.seekF = Math.max(a, Math.min(b - 1, f)); tlSeek(TLX.seekF); capture(); tlxRender(); return;
    }
    if (TLX.tool === "hand") { TLX.drag = { kind: "pan", x0: e.clientX, v0: w[0], v1: w[1] }; capture(); return; }
    if (TLX.tool === "zoom") { tlxZoom(e.altKey ? 2 : 0.5, fr); return; }
    if (TLX.tool === "razor") {
      const g = tlxSnapF(f, e.shiftKey);
      if (bar) { splitLayerAt(+bar.dataset.i, g); return; }
      if (g > a && g < b) addEdit(g);
      return;
    }
    if (bar) {                                       // adjustment layer: move (also to another track) / trim its ends
      const i = +bar.dataset.i, l = S.layers[i], r = bar.getBoundingClientRect(), edge = Math.min(7, r.width / 4);
      const kind = e.clientX - r.left < edge ? "l" : r.right - e.clientX < edge ? "r" : "move";
      TLX.drag = { kind, i, f0: f, s0: l.start, e0: l.end, t0: l.track || 1, moved: false };
      TLX.focus = "layer";
      if (S.layerSel !== i) { S.layerSel = i; sync(); }
      capture(); tlxRender(); return;
    }
    if (part) {                                      // a part of the episode (V1 / A1 / S): edges trim, a click selects
      const segs = epSegs(), pa = +part.dataset.a, pb = +part.dataset.b, i = segs.findIndex((s) => s.a === pa && s.b === pb);
      const r = part.getBoundingClientRect(), edge = Math.min(7, r.width / 3);
      const side = e.clientX - r.left < edge ? "a" : r.right - e.clientX < edge ? "b" : null;
      TLX.focus = "part";
      if (e.altKey && side) { joinEdit(side === "a" ? pa : pb); return; }
      if (!side || i < 0) { TLX.sel = { a: pa, b: pb }; TLX.drag = { kind: "partclick", moved: false }; capture(); tlxRender(); return; }
      let j = side === "a" ? i : i + 1;              // boundary index: between segs[j - 1] and segs[j]
      if (TLX.tool === "ripple" && j > 0 && j < segs.length) {
        const nb = side === "a" ? segs[i - 1] : segs[i + 1];
        if (!segs[i].cut && !nb.cut) {               // trimming into a kept neighbour: a new cut-out part grows there
          const at = side === "a" ? pa : pb;
          if (side === "a") { segs.splice(i, 0, { a: at, b: at, cut: true }); j = i + 1; }
          else { segs.splice(i + 1, 0, { a: at, b: at, cut: true }); j = i + 1; }
        }
      }
      TLX.drag = { kind: "edge", segs, j, side, ripple: TLX.tool === "ripple", grabbed: side === "a" ? j : j - 1, moved: false };
      TLX.sel = null; capture(); tlxRender(); return;
    }
    TLX.focus = null;                                // empty track: seek
    TLX.sel = null;
    TLX.drag = { kind: "seek" }; TLX.seekF = Math.max(a, Math.min(b - 1, f)); tlSeek(TLX.seekF); capture(); tlxRender();
  });
  body.addEventListener("pointermove", (e) => {
    if (!S.info) return;
    const fr = ltFrameRaw(e.clientX), f = Math.round(fr), d = TLX.drag, [a, b] = epRange();
    if (TLX.tool === "razor" && !d) {
      const g = tlxSnapF(f, e.shiftKey), rz = $("tlxRazor");
      rz.style.left = ltX(g, ltWindow()) + "%"; rz.classList.remove("hidden"); ltTip(tcf(g), e.clientX); return;
    }
    if (!d) { ltTip(`${tcf(f)} · frame ${f}`, e.clientX); return; }
    if (d.kind === "seek") { TLX.seekF = Math.max(a, Math.min(b - 1, f)); tlSeek(TLX.seekF); ltTip(tcf(TLX.seekF), e.clientX); return; }
    if (d.kind === "pan") {
      const r = $("ltBody").getBoundingClientRect(), df = (e.clientX - d.x0) / r.width * (d.v1 - d.v0);
      TLX.v0 = d.v0 - df; TLX.v1 = d.v1 - df; tlxClamp(); tlxRender(); return;
    }
    if (d.kind === "partclick") return;
    if (d.kind === "edge") {
      const segs = d.segs, j = d.j, g = tlxSnapF(f, e.shiftKey);
      d.moved = true;
      if (j === 0 || j === segs.length) { trimEpisodeEnd(j === 0 ? 0 : 1, g); tlxRender(); ltTip(`${j === 0 ? "In" : "Out"} ${tcf(g)}`, e.clientX); return; }
      let lo = segs[j - 1].a + 1, hi = segs[j].b - 1;
      if (d.ripple) { if (d.side === "b") hi = Math.min(hi, segs[j].b); else lo = Math.max(lo, segs[j - 1].a); }
      const v = Math.max(lo, Math.min(hi, g));
      segs[j - 1].b = v; segs[j].a = v;
      setSegs(segs); tlxRender();
      ltTip(`${tcf(v)} · ${d.ripple ? "ripple trim" : "edit moved"}`, e.clientX);
      return;
    }
    // adjustment layer drag (as before: snaps; another track when it is free there)
    const l = S.layers[d.i], minLen = 2;
    if (d.kind === "move") {
      const len = d.e0 - d.s0;
      let s = d.s0 + (f - d.f0);
      const s2 = tlxSnapF(s, e.shiftKey, l), e2 = tlxSnapF(s + len, e.shiftKey, l);
      s = s2 !== s ? s2 : e2 !== s + len ? e2 - len : s;
      s = Math.max(a, Math.min(b - len, s));
      const t = ltTrackAt(e.clientY);
      const lanesEls = $$("#ltLanes .lt-lane[data-track]"), line = tlxDropLine();
      d.newTrack = null; line.classList.add("hidden");
      if (t == null && lanesEls.length) {              // above the top track / below the bottom one: a new track there when you let go
        const top = lanesEls[0].getBoundingClientRect(), bot = lanesEls[lanesEls.length - 1].getBoundingClientRect(), bb = body.getBoundingClientRect();
        if (e.clientY < top.top - 2) d.newTrack = "above"; else if (e.clientY > bot.bottom + 6) d.newTrack = "below";
        if (d.newTrack) { line.style.top = ((d.newTrack === "above" ? top.top : bot.bottom) - bb.top - 1) + "px"; line.classList.remove("hidden"); }
      }
      l.start = s; l.end = s + len;
      if (t != null) l.track = t;                      // an occupied track too: what it overlaps moves away on release
    } else {
      const g = gapAt(l.track || 1, d.kind === "l" ? l.end - 1 : l.start, l) || [a, b];
      if (d.kind === "l") l.start = Math.max(g[0], Math.min(l.end - minLen, tlxSnapF(f, e.shiftKey, l)));
      else l.end = Math.min(g[1], Math.max(l.start + minLen, tlxSnapF(f, e.shiftKey, l)));
    }
    d.moved = true;
    ltTip(`${d.newTrack ? "new layer " + d.newTrack : "Layer " + (l.track || 1)} · ${tcf(l.start)} – ${tcf(l.end)} · ${l.end - l.start} fr`, e.clientX);
    tlxRender();
  });
  const end = () => {
    const d = TLX.drag; TLX.drag = null;
    setTimeout(() => $("ltTip").classList.add("hidden"), 600);
    if (!d) return;
    if (d.kind === "seek") { const f = TLX.seekF; TLX.seekF = null; if (f != null) ltSeekExact(f); return; }
    if (d.kind === "pan" || d.kind === "partclick") { tlxRender(); return; }
    if (d.kind === "edge") {
      if (d.j === 0 || d.j === d.segs.length) {       // the episode's In / Out moved: preview the new start / end
        S.frame = d.j === 0 ? 0 : null; S.total = 0;
        if (d.j !== 0) { $("rngPos").max = 1000; $("rngPos").value = 1000; }
        renderEpisodes(); trimInfo(); schedulePreview(); scheduleScript();
        log(`Episode ${S.current + 1}: ${tc(S.episodes[S.current][0])} – ${tc(S.episodes[S.current][1])}`, "ok");
      } else if (d.moved) { setSegs(d.segs.filter((s) => s.b > s.a)); editsChanged(); }
      else setSegs(d.segs.filter((s) => s.b > s.a));
      tlxRender(); return;
    }
    const dl = $("rgDropLine"); if (dl) dl.classList.add("hidden");
    if (d.moved) {
      const l = S.layers[d.i];
      if (l && d.kind === "move") {
        if (d.newTrack) l.track = d.newTrack === "above" ? S.tracks + 0.5 : 0.5;     // renumbered below
        resolveTracks(l); normalizeTracks();
      }
      layersChanged();
    } else selectLayer(d.i);
  };
  body.addEventListener("pointerup", end);
  body.addEventListener("pointercancel", end);
  body.addEventListener("pointerleave", () => { if (!TLX.drag) { $("ltTip").classList.add("hidden"); $("tlxRazor").classList.add("hidden"); } });
  body.addEventListener("contextmenu", (e) => {      // right-click on an adjustment layer: delete layer / track
    const bar = e.target.closest(".lt-layer:not(.rg-blk)");
    if (bar) layerMenu(e, +bar.dataset.i);
  });
  body.addEventListener("dblclick", (e) => {
    const cbar = e.target.closest(".lt-layer:not(.rg-blk)");
    if (cbar) { selectLayer(+cbar.dataset.i); goto("colour"); return; }       // a colour / LUT layer: edit it in the Colour tab
    const rbar = e.target.closest(".rg-blk");
    if (rbar) { goto("adjust"); return; }                                     // a filter's range: its controls are in Filtering
    if (TLX.tool !== "sel" || e.target.closest(".lt-layer") || e.target.closest(".part")) return;
    const t = ltTrackAt(e.clientY);
    if (t) addLayer(t, ltFrame(e.clientX));
  });
  $("tlxScrollV").addEventListener("wheel", (e) => {
    if (!S.info) return;
    if (e.ctrlKey || e.altKey) { e.preventDefault(); tlxZoom(e.deltaY < 0 ? 0.8 : 1.25, ltFrameRaw(e.clientX)); return; }
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.target.closest("#ltRuler")) {
      e.preventDefault();
      const w = ltWindow(), d = (e.deltaY + e.deltaX) / 800 * (w[1] - w[0]);
      TLX.v0 += d; TLX.v1 += d; tlxClamp(); tlxRender();
    }
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (!S.info || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) || document.activeElement.isContentEditable) return;
    const k = e.key.toLowerCase();
    if (e.ctrlKey && k === "k") {
      e.preventDefault();
      const layer = TLX.focus === "layer" || (S.step === "colour" && TLX.focus !== "part");    // Colour: Ctrl+K splits the layer, as before
      if (layer && S.layers.some((l) => tlxHead() > l.start && tlxHead() < l.end)) splitLayer(); else addEdit(Math.round(tlxHead()));
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      if (TLX.focus === "part" && TLX.sel) { e.preventDefault(); toggleCut(TLX.sel); return; }
      if ((TLX.focus === "layer" || S.step === "colour") && S.layers[S.layerSel]) { e.preventDefault(); removeLayer(S.layerSel); }
      return;
    }
    const tools = { v: "sel", c: "razor", r: "ripple", h: "hand", z: "zoom" };
    if (tools[k]) { tlxSetTool(tools[k]); return; }
    if (k === "q" || k === "w") { e.preventDefault(); rippleTo(k); return; }
    if (e.key === "=" || e.key === "+") { tlxZoom(0.5, tlxHead()); return; }
    if (e.key === "-" || e.key === "_") { tlxZoom(2, tlxHead()); return; }
    if (e.key === "\\") { tlxFit(); tlxRender(); return; }
    if (e.key === "ArrowUp" || e.key === "ArrowDown") { e.preventDefault(); tlxEditJump(e.key === "ArrowUp" ? -1 : 1); return; }
    if (e.key === "Home") { e.preventDefault(); togglePlay(false); gotoSrc(epRange()[0]); return; }
    if (e.key === "End") { e.preventDefault(); togglePlay(false); gotoSrc(epRange()[1] - 1); }
  });
  window.addEventListener("resize", tlxRender);
  new ResizeObserver(() => tlxRender()).observe(body);
}
function tlxEditJump(dir) {                           // ↑ / ↓: the edit point before / after the playhead
  const f = Math.round(tlxHead()), pts = [...new Set(epSegs().flatMap((s) => [s.a, s.b]))].sort((x, y) => x - y);
  const t = dir < 0 ? [...pts].reverse().find((p) => p < f - 0) : pts.find((p) => p > f);
  if (t == null) return;
  togglePlay(false); gotoSrc(Math.min(t, epRange()[1] - 1));
}
function initLayerTrack() { /* the Timeline (initTimeline) handles the layers too */ }
// keep the playhead in view while playing (pages on, like an editor)
function tlxFollow(src) {
  TLX.playSrc = src;
  const w = ltWindow();
  if (src != null && (src > w[1] || src < w[0])) { const sp = w[1] - w[0]; TLX.v0 = src - sp * 0.05; TLX.v1 = TLX.v0 + sp; tlxClamp(); }
  tlxRender();
}
