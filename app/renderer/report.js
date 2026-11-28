// Problems, logs and reports (loaded after app.js).
//  - a ring of the user's recent ACTIONS (clicks on named controls, shortcuts, requests to the engine: never typed text)
//  - every error the window or the engine reports is noted (and written to temp\evatoolkit.log by the main process)
//  - a "problems" button appears in the status bar; Help > Report a problem... makes ONE small zip (what was written, version
//    and system, logs, actions, errors, settings of the source, the script, a screenshot) that the user can send.
//  - the engine can be restarted (View > Restart engine; offered by itself when it stops).
"use strict";

const REP = { crumbs: [], errors: [], unseen: 0, lastRestart: 0, busyMaking: false };
function crumb(kind, text) {
  REP.crumbs.push({ t: Date.now(), k: kind, m: String(text).replace(/\s+/g, " ").slice(0, 220) });
  if (REP.crumbs.length > 400) REP.crumbs.shift();
}
function appLog(level, text) { try { window.api.appLog(level, String(text).slice(0, 4000)); } catch { /* no bridge */ } }
function noteError(kind, msg, stack) {
  if (window.__selftest) return;
  const e = { t: Date.now(), kind, msg: String(msg).slice(0, 600), stack: String(stack || "").slice(0, 1500) };
  REP.errors.push(e); if (REP.errors.length > 60) REP.errors.shift();
  REP.unseen++;
  appLog("error", `${kind}: ${e.msg}${e.stack ? "\n" + e.stack : ""}`);
  problemsUi();
}
function problemsUi() {
  const b = document.getElementById("btnProblems");
  if (!b) return;
  b.classList.toggle("hidden", REP.unseen === 0);
  b.querySelector("b").textContent = String(REP.unseen);
  b.querySelector("span").textContent = REP.unseen === 1 ? "problem" : "problems";
}

// ---- what the user did (names of controls, never what was typed)
function describe(el) {
  const t = el.closest("button, [data-cmd], [data-page], [data-dock], [data-left], [data-gid], .pick-card, .vk-card, .layer-row, summary, label, select, input, a");
  if (!t) return "";
  const isField = /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);
  const name = t.id || t.dataset.cmd || t.dataset.page || t.dataset.dock || t.dataset.gid || "";
  const label = isField ? (t.title || t.name || t.type || "") : (t.title || t.textContent || "");
  return `${t.tagName.toLowerCase()}${name ? "#" + name : ""} "${String(label).replace(/\s+/g, " ").trim().slice(0, 60)}"`;
}
document.addEventListener("click", (e) => { const d = describe(e.target); if (d) crumb("click", d); }, true);
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || /^(Delete|Escape|F\d+|Backspace)$/.test(e.key)) {
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (!typing || e.ctrlKey || e.metaKey) crumb("key", `${e.ctrlKey ? "Ctrl+" : ""}${e.shiftKey ? "Shift+" : ""}${e.altKey ? "Alt+" : ""}${e.key}`);
  }
}, true);

// ---- errors
window.addEventListener("error", (e) => noteError("error", e.message, `${String(e.filename || "").split("/").pop()}:${e.lineno}:${e.colno}\n${(e.error && e.error.stack) || ""}`));
window.addEventListener("unhandledrejection", (e) => noteError("rejection", (e.reason && e.reason.message) || String(e.reason), e.reason && e.reason.stack));
{
  const ce = console.error;
  console.error = (...a) => { noteError("console", a.map((x) => (x && x.stack) || String(x)).join(" ")); ce.apply(console, a); };
}

// ---- the app's own log / status / requests, noted
{
  const _log = window.log, _rpc = window.rpc;
  window.log = function (msg, cls) {
    if (cls === "err") { crumb("log-err", msg); noteError("log", msg); } else if (cls === "warn") crumb("log-warn", msg);
    return _log.apply(this, arguments);
  };
  const QUIET = new Set(["preview", "scopes", "vfr_auto", "locate", "stop_play", "my_nodes", "thumbs"]);
  window.rpc = async function (method, params) {
    const t0 = performance.now();
    try {
      const r = await _rpc.apply(this, arguments);
      const ms = Math.round(performance.now() - t0);
      if (!QUIET.has(method) || ms > 4000) crumb("rpc", `${method} ${ms} ms`);
      return r;
    } catch (e) {
      crumb("rpc-err", `${method} FAILED ${String((e && e.message) || e).slice(0, 160)}`);
      throw e;
    }
  };
}

// ---- the engine stopped: say so, and bring it back
async function restartEngine(auto) {
  REP.lastRestart = Date.now();
  appLog("info", `restarting the engine (${auto ? "automatic" : "asked"})`);
  status("Restarting the engine...");
  try { await window.api.restartEngine(); } catch { /* shown below */ }
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 400));
    try { await window.api.rpc("hello", {}).then((r) => { if (r && r.error) throw new Error(r.error); }); S.busy = false; updateButtons(); status("The engine is running again.", "ok"); log("The engine was restarted.", "ok"); schedulePreview(); return true; } catch { /* still starting */ }
  }
  status("The engine did not start again - reload the window (Ctrl+R) or report the problem.", "err");
  return false;
}
window.engineStopped = function (code) {
  const msg = `The engine stopped (exit code ${code}).`;
  log(msg, "err"); status(msg + " Restarting it...", "err");
  if (Date.now() - REP.lastRestart > 60000) restartEngine(true);
  else status(msg + " View > Restart engine, or Help > Report a problem.", "err");
};

// ---- the report
function replyList(box, items) { box.textContent = ""; for (const t of items) box.appendChild(Object.assign(document.createElement("div"), { textContent: t })); }
function reportOpen() {
  const dlg = document.getElementById("reportDlg");
  dlg.classList.remove("hidden");
  document.getElementById("repDone").classList.add("hidden");
  const box = document.getElementById("repProblems");
  const recent = REP.errors.slice(-5).reverse().map((e) => `${new Date(e.t).toLocaleTimeString()}  ${e.kind}: ${e.msg.slice(0, 140)}`);
  box.classList.toggle("hidden", !recent.length);
  if (recent.length) { replyList(box, recent); box.prepend(Object.assign(document.createElement("b"), { textContent: "Recent problems that will be in the report:" })); }
  REP.unseen = 0; problemsUi();
  document.getElementById("repText").focus();
}
function reportClose() { document.getElementById("reportDlg").classList.add("hidden"); }
async function reportMake() {
  if (REP.busyMaking) return;
  REP.busyMaking = true;
  const btn = document.getElementById("btnReportMake"), done = document.getElementById("repDone");
  btn.disabled = true; btn.textContent = "Creating…";
  try {
    const on = (id) => document.getElementById(id).checked;
    let shot = null;
    if (on("repShot")) { document.getElementById("reportDlg").classList.add("hidden"); await new Promise((r) => setTimeout(r, 250)); shot = await window.api.captureWindow(); document.getElementById("reportDlg").classList.remove("hidden"); }
    const sys = await window.api.systemInfo();
    const withSettings = on("repSet") && S.info;
    const payload = {
      text: document.getElementById("repText").value, hide_names: on("repHide"), sysinfo: sys, page: S.step || "", log_file: sys.logFile,
      actions: on("repLog") ? REP.crumbs.map((c) => `${new Date(c.t).toISOString().slice(11, 23)} ${c.k.padEnd(8)} ${c.m}`).join("\n") : "",
      errors: REP.errors.map((e) => `${new Date(e.t).toISOString().slice(11, 23)} ${e.kind}: ${e.msg}\n${e.stack}`).join("\n\n"),
      ui_log: on("repLog") ? document.getElementById("log").innerText.slice(-30000) : "",
      with_logs: on("repLog"), with_source_info: on("repSrc") && !!S.info,
      state: withSettings ? backendState([S.current]) : null,
      script: withSettings ? (S.script || (await window.rpc("script", { state: backendState([S.current]), episode: S.current }).catch(() => "")) || "") : "",
      screenshot: shot, version: (document.querySelector(".brand .ver") || {}).textContent || "",
    };
    const r = await window.rpc("make_report", payload);
    appLog("info", `report created: ${r.path} (${r.mb} MB)`);
    done.classList.remove("hidden"); done.classList.remove("err");
    done.innerHTML = "";
    done.append(`Report saved (${r.mb} MB): `, Object.assign(document.createElement("code"), { textContent: r.path }), document.createElement("br"), "Send this zip file with your message. It contains: " + r.files.join(", ") + ".");
    const row = document.createElement("div"); row.className = "rep-done-btns";
    const show = Object.assign(document.createElement("button"), { textContent: "Show in folder" }); show.onclick = () => window.api.showItem(r.path);
    const copy = Object.assign(document.createElement("button"), { textContent: "Copy the file path" }); copy.onclick = () => navigator.clipboard.writeText(r.path).catch(() => {});
    row.append(show, copy); done.appendChild(row);
    window.api.showItem(r.path);
  } catch (e) {
    done.classList.remove("hidden"); done.classList.add("err"); done.textContent = "The report could not be made: " + ((e && e.message) || e) + " - Help > Open logs folder has the log file to send instead.";
  } finally { REP.busyMaking = false; btn.disabled = false; btn.textContent = "Create report"; }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btnProblems").onclick = reportOpen;
  document.getElementById("btnReportClose").onclick = reportClose;
  document.getElementById("btnReportMake").onclick = reportMake;
  document.getElementById("btnReportLogs").onclick = () => window.api.openLogs();
  document.getElementById("reportDlg").onclick = (e) => { if (e.target.id === "reportDlg") reportClose(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !document.getElementById("reportDlg").classList.contains("hidden")) reportClose(); });
  Object.assign(WS_CMD, { report: reportOpen, "open-logs": () => window.api.openLogs(), "engine-restart": () => restartEngine(false) });
  crumb("start", `window loaded ${document.querySelector(".brand .ver") ? document.querySelector(".brand .ver").textContent : ""}`);
});
