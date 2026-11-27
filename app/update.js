// EVA toolkit - updates: the latest GitHub release of the app, downloaded and installed over this portable folder.
// Check: one public request to api.github.com (no account, nothing about the user is sent). Install: the release zip
// is unpacked into update\staged, checked, and copied over the app folder by a small script once the app has closed
// (it only adds and replaces files: runtime\python, engines\, temp\, settings.json and your own filters stay).
const { net } = require("electron");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const REPO = "EVA0111/DVDRestore";
const API = `https://api.github.com/repos/${REPO}/releases/latest`;

// versions are counted as decimals, as the releases are numbered: 0.15 < 0.2 < 0.25 < 0.3 (then a patch number)
function parseVer(s) {
  const m = String(s || "").match(/(\d+)\.(\d+)(?:\.(\d+))?/);
  return m ? [parseFloat(`${m[1]}.${m[2]}`), Number(m[3] || 0)] : null;
}
function isNewer(latest, current) {
  const a = parseVer(latest), b = parseVer(current);
  if (!a || !b) return false;
  return a[0] > b[0] + 1e-9 || (Math.abs(a[0] - b[0]) < 1e-9 && a[1] > b[1]);
}
const pretty = (v) => { const p = parseVer(v); return p ? `v${p[0]}${p[1] ? "." + p[1] : ""}` : String(v); };

function getJSON(url, version) {
  return new Promise((resolve, reject) => {
    const req = net.request({ url, redirect: "follow" });
    req.setHeader("User-Agent", `EVAtoolkit/${version}`);
    req.setHeader("Accept", "application/vnd.github+json");
    let body = "";
    req.on("response", (res) => {
      res.on("data", (b) => { body += b; });
      res.on("end", () => {
        if (res.statusCode === 404) return reject(new Error("no release published yet"));
        if (res.statusCode >= 400) return reject(new Error(`GitHub answered HTTP ${res.statusCode}`));
        try { resolve(JSON.parse(body)); } catch (e) { reject(e); }
      });
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end();
  });
}

async function check(current) {
  const r = await getJSON(API, current);
  const zip = (r.assets || []).find((a) => /\.zip$/i.test(a.name));
  return {
    current: pretty(current), latest: pretty(r.tag_name), tag: r.tag_name, name: r.name || r.tag_name,
    newer: isNewer(r.tag_name, current) && !r.draft, notes: String(r.body || "").slice(0, 4000), page: r.html_url,
    asset: zip ? { name: zip.name, url: zip.browser_download_url, size: zip.size } : null,
  };
}

function findRoot(dir, depth = 0) {                    // the unpacked folder that holds app\ and core\
  if (fs.existsSync(path.join(dir, "app", "package.json")) && fs.existsSync(path.join(dir, "core"))) return dir;
  if (depth > 2) return null;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) { const r = findRoot(path.join(dir, e.name), depth + 1); if (r) return r; }
  }
  return null;
}

// download + unpack + check; returns the folder to install from
async function prepare(appDir, info, download, onProgress, signal) {
  if (!info.asset) throw new Error("this release has no zip to download");
  const up = path.join(appDir, "update");
  const staged = path.join(up, "staged");
  fs.rmSync(staged, { recursive: true, force: true });
  fs.mkdirSync(staged, { recursive: true });
  const zip = path.join(up, info.asset.name);
  if (!(fs.existsSync(zip) && fs.statSync(zip).size === info.asset.size)) {
    await download(info.asset.url, zip, (got, total) => onProgress("download", got, total || info.asset.size), signal);
  }
  onProgress("unpack", 0, 1);
  await new Promise((resolve, reject) => {           // Windows 10+ tar.exe reads zip files
    // Windows' own tar (bsdtar) by full path: a "tar" found on PATH can be Git's GNU tar, which reads "D:" as a host
    const tar = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");
    const p = spawn(tar, ["-xf", zip, "-C", staged], { windowsHide: true });
    if (signal) signal.onCancel = () => { try { p.kill(); } catch {} };
    let err = "";
    p.stderr.on("data", (b) => { err += b; });
    p.on("error", reject);
    p.on("exit", (c) => (c === 0 ? resolve() : reject(new Error("unpacking failed: " + err.slice(-300)))));
  });
  const root = findRoot(staged);
  if (!root) throw new Error("the downloaded zip does not contain the app (app\\ and core\\)");
  const v = JSON.parse(fs.readFileSync(path.join(root, "app", "package.json"), "utf-8")).version;
  const exe = fs.readdirSync(root).find((f) => /\.exe$/i.test(f));
  onProgress("ready", 1, 1);
  return { root, version: v, exe };
}

// the copy runs after this process has exited; then the new version starts
function install(appDir, prepared, pid) {
  const up = path.join(appDir, "update");
  const log = path.join(up, "apply.log");
  const exe = prepared.exe ? path.join(appDir, prepared.exe) : "";
  const cmd = [
    "@echo off",
    "chcp 65001 >nul",
    `echo EVA toolkit update to ${prepared.version} > "${log}"`,
    ":wait",
    `tasklist /FI "PID eq ${pid}" 2>nul | find "${pid}" >nul && (timeout /t 1 /nobreak >nul & goto wait)`,
    "timeout /t 2 /nobreak >nul",
    // /E: add and replace only - nothing of yours is removed; retries ride out files the closing app still holds
    `robocopy "${prepared.root}" "${appDir}" /E /R:20 /W:1 /NP /NFL /NDL >> "${log}"`,
    `if %ERRORLEVEL% GEQ 8 (echo COPY FAILED >> "${log}" & start "" notepad "${log}" & exit /b 1)`,
    `rmdir /s /q "${path.join(up, "staged")}"`,
    `del /q "${path.join(up, "*.zip")}" 2>nul`,                   // the downloaded zip (300 MB) is no longer needed
    exe ? `start "" "${exe}"` : "",
  ].filter(Boolean).join("\r\n") + "\r\n";
  const bat = path.join(up, "apply.cmd");
  fs.writeFileSync(bat, cmd, "utf-8");
  const p = spawn("cmd.exe", ["/c", bat], { detached: true, stdio: "ignore", windowsHide: true });
  p.unref();
}

module.exports = { check, prepare, install, isNewer, parseVer };
