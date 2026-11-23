// DVD Restore - Electron main process.
// Starts the Python backend (../core/server.py), relays JSON-line RPC between it and the window,
// and provides native dialogs.
const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const readline = require("readline");

const deps = require("./deps");

const APP_DIR = path.resolve(__dirname, "..");                 // project root: app/ core/ models/ runtime/
const SERVER = path.join(APP_DIR, "core", "server.py");
// the shared (portable) copy has portable.json at its root: it only uses what is in its own runtime\ folder and
// installs that in the Dependencies window. The development folder uses the installed Python / tools as before.
const PORTABLE = fs.existsSync(path.join(APP_DIR, "portable.json"));
const ICON = path.join(__dirname, "icon.png");
app.setAppUserModelId("DVDRestore");                           // taskbar: own group + icon, not Electron's

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

let win = null;
let py = null;
let nextId = 1;
const pending = new Map();

function startBackend() {
  py = spawn(findPython(), ["-I", "-W", "ignore", SERVER], {
    cwd: APP_DIR,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONNOUSERSITE: "1" },
    windowsHide: true,
  });
  readline.createInterface({ input: py.stdout }).on("line", (line) => {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.id !== undefined && pending.has(msg.id)) {
      const { resolve } = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(msg);
    } else if (msg.event && win) {
      win.webContents.send("backend-event", msg);
    }
  });
  readline.createInterface({ input: py.stderr }).on("line", (line) => {
    if (win && line && !line.includes("API 3") && !line.includes("API3")) {
      win.webContents.send("backend-event", { event: "stderr", msg: line });
    }
  });
  py.on("exit", (code) => {
    for (const { resolve } of pending.values()) resolve({ error: "backend stopped" });
    pending.clear();
    if (win) win.webContents.send("backend-event", { event: "exit", code });
  });
}

ipcMain.handle("rpc", (_e, method, params) => new Promise((resolve) => {
  if (!py || py.exitCode !== null) return resolve({ error: "backend not running" });
  const id = nextId++;
  pending.set(id, { resolve });
  py.stdin.write(JSON.stringify({ id, method, params: params || {} }) + "\n");
}));

ipcMain.handle("open-video", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Open video or DVD image",
    properties: ["openFile"],
    filters: [{ name: "Video, DVD or image", extensions: ["mkv", "iso", "m2v", "vob", "mpg", "ts", "mp4", "mov", "avi", "webm", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] },
              { name: "Images", extensions: ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"] },
              { name: "All files", extensions: ["*"] }],
  });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle("open-lut", async () => {
  const r = await dialog.showOpenDialog(win, {
    title: "Add LUTs (.cube)", properties: ["openFile", "multiSelections"],
    filters: [{ name: "LUT", extensions: ["cube"] }, { name: "All files", extensions: ["*"] }],
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
ipcMain.handle("show-path", (_e, p) => shell.openPath(p));
const GITHUB_URL = "https://github.com/EVA0111";                 // the GitHub button (fixed address, nothing else opens)
ipcMain.handle("open-github", () => shell.openExternal(GITHUB_URL));
ipcMain.handle("show-item", (_e, p) => shell.showItemInFolder(p));
ipcMain.handle("is-dir", (_e, p) => { try { return fs.statSync(p).isDirectory(); } catch { return false; } });

function createWindow() {
  win = new BrowserWindow({
    width: 1600, height: 960, minWidth: 1280, minHeight: 720,
    backgroundColor: "#15171c", icon: ICON, title: "DVD Restore v0.1 alpha", autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  win.on("closed", () => { win = null; });
}

// ---------------------------------------------------------------- dependencies window
let depsWin = null;
let depsJob = null;                                            // the running install (cancel signal)
let started = false;

function openDeps(firstRun) {
  if (depsWin) { depsWin.focus(); return; }
  depsWin = new BrowserWindow({
    width: 760, height: 640, minWidth: 620, minHeight: 520, parent: firstRun ? undefined : (win || undefined),
    backgroundColor: "#15171c", icon: ICON, title: "DVD Restore v0.1 alpha - Dependencies", autoHideMenuBar: true,
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
ipcMain.handle("deps-start", () => {                           // "Start DVD Restore" after the first install
  if (!started) { started = true; startBackend(); createWindow(); }
  if (depsWin) depsWin.close();
});

app.whenReady().then(() => {
  deps.syncLayout(APP_DIR);                                    // engines/ temp/ runtime/ made again if deleted
  if (PORTABLE && !deps.check(APP_DIR).ready) { openDeps(true); return; }
  started = true; startBackend(); createWindow();
});
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
