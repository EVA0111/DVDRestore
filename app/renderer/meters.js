// Status bar meters: CPU, RAM, GPU load and VRAM (main.js "sys-stats": os.cpus deltas + nvidia-smi), every 1.5 s
// while the window is shown. Green / amber / red at 70 % and 90 %.
"use strict";

const METERS = { timer: 0, built: false };
function meterEl(id, label, wide) {
  const m = el("div", "meter" + (wide ? " wide" : "")); m.id = id;
  m.append(el("b", null, label), el("i"), el("span", null, "–"));
  return m;
}
function meterSet(id, pct, text, tip) {
  const m = $(id);
  if (!m) return;
  m.classList.remove("hidden");
  m.querySelector("i").style.setProperty("--v", Math.max(0, Math.min(100, pct || 0)) + "%");
  m.querySelector("span").textContent = text;
  m.classList.toggle("mid", pct >= 70 && pct < 90);
  m.classList.toggle("high", pct >= 90);
  if (tip) m.title = tip;
}
const gb = (b) => (b / 1073741824).toFixed(1);
async function metersTick() {
  clearTimeout(METERS.timer);
  if (!document.hidden && window.api.sysStats) {
    try {
      const s = await window.api.sysStats();
      if (!METERS.built) {
        const box = $("meters");
        box.append(meterEl("mCpu", "CPU"), meterEl("mRam", "RAM", true));
        if (s.gpu_name) box.append(meterEl("mGpu", "GPU"), meterEl("mVram", "VRAM", true));
        METERS.built = true;
      }
      if (s.cpu != null) meterSet("mCpu", s.cpu, `${Math.round(s.cpu)} %`, `CPU: ${Math.round(s.cpu)} % of ${s.cores} threads`);
      meterSet("mRam", 100 * s.ram_used / s.ram_total, `${gb(s.ram_used)} / ${gb(s.ram_total)} GB`, `Memory in use (the whole computer): ${gb(s.ram_used)} of ${gb(s.ram_total)} GB`);
      if (s.gpu_name) {
        meterSet("mGpu", s.gpu, `${Math.round(s.gpu)} %`, `${s.gpu_name}: ${Math.round(s.gpu)} % busy${isFinite(s.gpu_temp) ? ` · ${Math.round(s.gpu_temp)} °C` : ""}`);
        meterSet("mVram", 100 * s.vram_used / s.vram_total, `${gb(s.vram_used)} / ${gb(s.vram_total)} GB`,
          `Video memory in use (all programs): ${gb(s.vram_used)} of ${gb(s.vram_total)} GB on the ${s.gpu_name}. TensorRT engines and the AI models live here.`);
      }
    } catch { /* main process busy: next time */ }
  }
  METERS.timer = setTimeout(metersTick, 1500);
}
document.addEventListener("DOMContentLoaded", () => { metersTick(); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) metersTick(); });
