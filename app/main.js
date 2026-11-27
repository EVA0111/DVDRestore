// EVA toolkit - Electron main process.
// Starts the Python backend (../core/server.py), relays JSON-line RPC between it and the window,
// and provides native dialogs.
const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const readline = require("readline");

const deps = require("./deps");
const update = require("./update");

const APP_DIR = path.resolve(__dirname, "..");                 // project root: app/ core/ models/ runtime/
const SERVER = path.join(APP_DIR, "core", "server.py");
// the shared (portable) copy has portable.json at its root: it only uses what is in its own runtime\ folder and
// installs that in the Dependencies window. The development folder uses the installed Python / tools as before.
const PORTABLE = fs.existsSync(path.join(APP_DIR, "portable.json"));
const ICON = path.join(__dirname, "icon.png");
app.setAppUserModelId("EVAtoolkit");                           // taskbar: own group + icon, not Electron's

function findPython() {
  // the portable version's own Python first, then an override, then a normal install
  if (PORTABLE) return path.join(APP_DIR, "runtime", "python", "python.exe");
  const candidates = [
    path.join(APP_DIR, "runtime", "python", "python.exe"),
    process.env.DVDRESTORE_PYTHON,
    path.join(process.env.LOCALAPPDATA || "", "Programs", "Python", "Python312", "python.exe"),
    "python",
  ].filter(Boolean);
  for (const c of candidates) {
    if (c === "python" || fs.existsSync(c)) return c;
  }
  return "python";
}

// ---------------------------------------------------------------- diagnostics log (temp\evatoolkit.log)
// Engine errors, failed requests, slow previews, engine exits, window errors / freezes - so a failure the user sees
// can be traced afterwards. Rolls over at 5 MB (one old copy, .1).
const LOG = path.join(APP_DIR, "temp", "evatoolkit.log");
let logWrites = 0;
function logf(...a) {
  try {
    if (logWrites++ % 200 === 0) {
      fs.mkdirSync(path.dirname(LOG), { recursive: true });
      if (fs.existsSync(LOG) && fs.statSync(LOG).size > 5e6) fs.renameSync(LOG, LOG + ".1");
    }
    fs.appendFileSync(LOG, `${new Date().toISOString()} ${a.join(" ")}\n`);
  } catch { /* logging must never break the app */ }
}

let win = null;
let py = null;
let nextId = 1;
const pending = new Map();

function startBackend() {
  logf(`--- start ${app.getVersion()} backend ${findPython()} portable=${PORTABLE}`);
  py = spawn(findPython(), ["-I", "-W", "ignore", SERVER], {
    cwd: APP_DIR,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONNOUSERSITE: "1" },
    windowsHide: true,
  });
  readline.createInterface({ input: py.stdout }).on("line", (line) => {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.id !== undefined && pending.has(msg.id)) {
      const { resolve, method, t0 } = pending.get(msg.id);
      pending.delete(msg.id);
      const ms = Date.now() - t0;
      if (msg.error && msg.error !== "superseded") logf(`RPC ${method} FAILED after ${ms} ms: ${String(msg.error).slice(0, 4000)}`);
      else if (ms > 15000 && (method === "preview" || method === "probe")) logf(`RPC ${method} slow: ${ms} ms`);
      resolve(msg);
    } else if (msg.event && win) {
      win.webContents.send("backend-event", msg);
    }
  });
  readline.createInterface({ input: py.stderr }).on("line", (line) => {
    if (line && !line.includes("API 3") && !line.includes("API3")) {
      if (!/DeprecationWarning|UserWarning/.test(line)) logf("ENGINE", line.slice(0, 4000));
      if (win) win.webContents.send("backend-event", { event: "stderr", msg: line });
    }
  });
  py.on("exit", (code, signal) => {
    logf(`ENGINE EXITED code=${code} signal=${signal} (pending requests: ${[...pending.values()].map((p) => p.method).join(", ") || "none"})`);
    for (const { resolve } of pending.values()) resolve({ error: "backend stopped" });
    pending.clear();
    if (win) win.webContents.send("backend-event", { event: "exit", code });
  });
}

ipcMain.handle("rpc", (_e, method, params) => new Promise((resolve) => {
  if (!py || py.exitCode !== null) return resolve({ error: "backend not running" });
  const id = nextId++;
  pending.set(id, { resolve, method, t0: Date.now() });
  py.stdin.write(JSON.stringify({ id, method, params: params || {} }) + "\n");
}));

ipcMain.handle("open-video", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Open video or DVD image",
    properties: ["openFile"],
    filters: [{ name: "Video, DVD or image", extensions: ["mkv", "iso", "m2v", "vob", "mpg", "ts", "m2ts", "mts", "mpls", "mp4", "mov", "avi", "webm", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] },
              { name: "Images", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] },
              { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle("open-refs", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Colour reference: a better release, screenshots or a good episode of the same show",
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Images or video", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "mkv", "mp4", "m2v", "vob", "mpg", "ts", "mov", "avi", "webm"] },
              { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? null : r.filePaths;
});
ipcMain.handle("open-lut", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Add LUTs (.cube)", properties: ["openFile", "multiSelections"],
    filters: [{ name: "LUT", extensions: ["cube"] }, { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? [] : r.filePaths;
});
ipcMain.handle("open-my-filter", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Add your own filters", properties: ["openFile", "multiSelections"],
    filters: [{ name: "Python filters or VapourSynth plugins", extensions: ["py", "dll"] }, { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? [] : r.filePaths;
});
ipcMain.handle("open-model", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Add AI models", properties: ["openFile", "multiSelections"],
    filters: [{ name: "AI model", extensions: ["onnx", "pth", "safetensors"] }, { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? [] : r.filePaths;
});
ipcMain.handle("open-folder", async (_e, title) => {
  const r = await dialog.showOpenDialog(win, { title: title || "Choose folder", properties: ["openDirectory"] });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle("save-text", async (_e, name, text, filters) => {
  const r = await dialog.showSaveDialog(win, { title: "Save script", defaultPath: name, filters: filters || [] });
  if (r.canceled || !r.filePath) return null;
  fs.writeFileSync(r.filePath, text, "utf-8");
  return r.filePath;
});
ipcMain.handle("open-text", async (_e, filters) => {
  const r = await dialog.showOpenDialog(win, { title: "Open", properties: ["openFile"], filters: [...(filters || []), { name: "All files", extensions: ["*"] }] });
  if (r.canceled || !r.filePaths[0]) return null;
  const st = fs.statSync(r.filePaths[0]);
  if (st.size > 20 * 1024 * 1024) return null;            // not a settings file
  return { path: r.filePaths[0], text: fs.readFileSync(r.filePaths[0], "utf-8") };
});
ipcMain.handle("show-path", (_e, p) => shell.openPath(p));
const GITHUB_URL = "https://github.com/EVA0111";                 // the GitHub button (fixed address, nothing else opens)
ipcMain.handle("open-github", () => shell.openExternal(GITHUB_URL));
ipcMain.handle("show-item", (_e, p) => shell.showItemInFolder(p));
ipcMain.handle("is-dir", (_e, p) => { try { return fs.statSync(p).isDirectory(); } catch { return false; } });

function createWindow() {
  win = new BrowserWindow({
    width: 1600, height: 960, minWidth: 1280, minHeight: 720,
    backgroundColor: "#15171c", icon: ICON, title: "EVA toolkit v0.25 alpha", autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  win.on("closed", () => { win = null; });
  // window-side problems: script errors, freezes, a crashed renderer
  win.webContents.on("console-message", (e) => {
    const lvl = e.level ?? e.params?.level;
    if (lvl === "error" || lvl === 3) logf("WINDOW", String(e.message ?? e.params?.message).slice(0, 2000));
  });
  win.on("unresponsive", () => logf("WINDOW unresponsive (frozen)"));
  win.on("responsive", () => logf("WINDOW responsive again"));
  win.webContents.on("render-process-gone", (_e, d) => logf(`WINDOW renderer gone: ${d.reason} (exit ${d.exitCode})`));
}

// ---------------------------------------------------------------- updates (update.js)
let updInfo = null, updPrepared = null, updJob = null;
const sendUpd = (msg) => { if (win) win.webContents.send("update-event", msg); };
ipcMain.handle("update-check", async () => {
  try { updInfo = await update.check(app.getVersion()); return { ...updInfo, portable: PORTABLE }; }
  catch (e) { return { error: String(e.message || e), portable: PORTABLE }; }
});
ipcMain.handle("update-download", async () => {
  if (!updInfo || !updInfo.newer) return { error: "no newer version to download" };
  if (updJob) return { error: "already downloading" };
  updJob = { cancelled: false };
  try {
    updPrepared = await update.prepare(APP_DIR, updInfo, deps.download,
      (stage, got, total) => sendUpd({ stage, got, total }), updJob);
    logf(`update ${updPrepared.version} ready in ${updPrepared.root}`);
    return { ok: true, version: updPrepared.version };
  } catch (e) {
    return { error: updJob.cancelled ? "cancelled" : String(e.message || e) };
  } finally { updJob = null; }
});
ipcMain.handle("update-cancel", () => { if (updJob) { updJob.cancelled = true; updJob.onCancel && updJob.onCancel(); } return true; });
ipcMain.handle("update-install", () => {
  if (!updPrepared) return { error: "nothing downloaded" };
  if (!PORTABLE) return { error: "this copy runs from the source folder - updates install only into the portable app" };
  logf(`installing update ${updPrepared.version}`);
  update.install(APP_DIR, updPrepared, process.pid);
  setTimeout(() => app.quit(), 300);                          // the script waits for this process, copies, restarts
  return { ok: true };
});
ipcMain.handle("open-release", () => { if (updInfo && /^https:\/\/github\.com\//.test(updInfo.page)) shell.openExternal(updInfo.page); });

// ---------------------------------------------------------------- dependencies window
let depsWin = null;
let depsJob = null;                                            // the running install (cancel signal)
let started = false;

function openDeps(firstRun) {
  if (depsWin) { depsWin.focus(); return; }
  depsWin = new BrowserWindow({
    width: 760, height: 640, minWidth: 620, minHeight: 520, parent: firstRun ? undefined : (win || undefined),
    backgroundColor: "#15171c", icon: ICON, title: "EVA toolkit v0.25 alpha - Dependencies", autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false },
  });
  depsWin.loadFile(path.join(__dirname, "renderer", "deps.html"), { query: { first: firstRun ? "1" : "0" } });
  depsWin.on("closed", () => {
    depsWin = null;
    if (depsJob) { depsJob.cancelled = true; depsJob.onCancel && depsJob.onCancel(); }
    if (!started && !win) app.quit();                          // closed on first run without starting
  });
}

const sendDeps = (msg) => { if (depsWin) depsWin.webContents.send("deps-event", msg); };
ipcMain.handle("deps-check", () => ({ ...deps.check(APP_DIR), portable: PORTABLE }));
ipcMain.handle("deps-install", async (_e, opts) => {
  if (depsJob) return { error: "already installing" };
  depsJob = { cancelled: false };
  const job = depsJob;
  try {
    await deps.install(APP_DIR, opts || {}, {
      status: (id, state) => sendDeps({ id, state }),
      progress: (id, done, total, file) => sendDeps({ id, done, total, file }),
      log: (id, line) => sendDeps({ id, line }),
      elapsed: (id, secs) => sendDeps({ id, elapsed: secs }),
    }, job);
    return { ok: true, ...deps.check(APP_DIR) };
  } catch (e) {
    return { error: job.cancelled ? "cancelled" : String(e.message || e) };
  } finally { depsJob = null; }
});
ipcMain.handle("deps-cancel", () => { if (depsJob) { depsJob.cancelled = true; depsJob.onCancel && depsJob.onCancel(); } });
ipcMain.handle("deps-folder", () => shell.openPath(path.join(APP_DIR, "runtime")));
ipcMain.handle("deps-open", () => openDeps(false));
ipcMain.handle("deps-start", () => {                           // "Start EVA toolkit" after the first install
  if (!started) { started = true; startBackend(); createWindow(); }
  if (depsWin) depsWin.close();
});

// One EVA toolkit at a time: two instances share the profile folder (AppData\Roaming\EVA toolkit) and the second
// one crashed / froze the first (seen 2026-11-25: a renderer crash the second another copy started). Starting it
// again brings the open window to the front. Test runs pass their own --user-data-dir, which has its own lock.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  logf("second copy started: the open EVA toolkit window is brought to the front instead");
  app.quit();
} else {
  app.on("second-instance", () => {
    const w = win || depsWin;
    if (w) { if (w.isMinimized()) w.restore(); w.show(); w.focus(); }
  });
  app.whenReady().then(() => {
    deps.syncLayout(APP_DIR);                                  // engines/ temp/ runtime/ made again if deleted
    if (PORTABLE && !deps.check(APP_DIR).ready) { openDeps(true); return; }
    started = true; startBackend(); createWindow();
  });
}
let stdinClosed = false;
function stopBackend() {
  // closing stdin lets the backend cancel playback / a running encode and exit; kill() is the fallback
  if (py && py.exitCode === null) { stdinClosed = true; try { py.stdin.end(); } catch {} setTimeout(() => { try { if (py && py.exitCode === null) py.kill(); } catch {} }, 1500); }
}
app.on("window-all-closed", () => { if (depsJob) return; stopBackend(); app.quit(); });
app.on("will-quit", stopBackend);
// last resort (the app is going down without the steps above): a backend whose stdin was closed is already
// stopping its jobs and exits by itself - killing it here would orphan a running vspipe / ffmpeg
process.on("exit", () => { try { if (py && py.exitCode === null && !stdinClosed) py.kill(); } catch {} });
