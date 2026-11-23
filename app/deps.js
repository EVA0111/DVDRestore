// DVD Restore - dependencies: what the engine needs next to the app, where it comes from, and the installer.
// Everything goes into <root>\runtime\ (nothing outside the folder, no registry, no PATH): the copy keeps working
// when the folder is moved or put on another drive. Downloads come from the official places only:
//   Python 3.12 (python.org's NuGet package), PyTorch (download.pytorch.org), TensorRT / VapourSynth + plugins (PyPI),
//   ffmpeg (gyan.dev release build), MKVToolNix (mkvtoolnix.download).
const { net } = require("electron");
const { spawn, execFile } = require("child_process");
const crypto = require("crypto");
const path = require("path");
const fs = require("fs");

const TAR = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");   // unpacks .zip and .7z

function layout(root) {
  const rt = path.join(root, "runtime");
  return {
    root, rt,
    python: path.join(rt, "python", "python.exe"),
    ffmpeg: path.join(rt, "ffmpeg", "ffmpeg.exe"),
    ffprobe: path.join(rt, "ffmpeg", "ffprobe.exe"),
    stampFf: path.join(rt, "ffmpeg", "dvdrestore-ffmpeg.txt"),
    mkvmerge: path.join(rt, "mkvtoolnix", "mkvmerge.exe"),
    downloads: path.join(rt, "downloads"),
    req: path.join(root, "core", "requirements.txt"),
    reqView: path.join(root, "core", "requirements-vsview.txt"),
    stamp: path.join(rt, "python", "dvdrestore-packages.txt"),
    stampView: path.join(rt, "python", "dvdrestore-vsview.txt"),
  };
}

const COMPONENTS = [
  { id: "python", name: "Python 3.12", what: "runs the engine", size: 15e6, required: true,
    url: "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg", file: "python-3.12.10.nupkg" },
  { id: "packages", name: "PyTorch, TensorRT, VapourSynth + plugins", size: 5.3e9, required: true,
    what: "the neural networks (CUDA 13), the GPU engines and the video filters - most of the download" },
  // the "full" build: the "essentials" one has no DVD-Video reader (libdvdnav), so .iso / VIDEO_TS would show no titles
  { id: "ffmpeg", name: "ffmpeg", what: "reads discs (DVD-Video) and files, encodes the result", size: 76e6, required: true,
    url: "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.1-full_build.7z", file: "ffmpeg-9.1-full_build.7z",
    fallback: "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z" },
  { id: "mkvtoolnix", name: "MKVToolNix", what: "puts video, audio and subtitles together (.mkv)", size: 31e6, required: true,
    url: "https://mkvtoolnix.download/windows/releases/103.0/mkvtoolnix-64-bit-103.0.7z", file: "mkvtoolnix-103.0.7z" },
  { id: "vsview", name: "vsview (optional)", what: "frame-accurate preview window for the Script step", size: 700e6, required: false },
];

function hashFile(p) {
  try { return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex").slice(0, 16); } catch { return ""; }
}
function readStamp(p) { try { return fs.readFileSync(p, "utf-8").trim(); } catch { return ""; } }

// state of every component: "ok", "missing" or "update" (the package list changed since it was installed)
function check(root) {
  const L = layout(root);
  const st = {};
  st.python = fs.existsSync(L.python) ? "ok" : "missing";
  const want = hashFile(L.req), have = readStamp(L.stamp);
  st.packages = st.python !== "ok" || !have ? "missing" : (have === want ? "ok" : "update");
  // the stamp names the build that was installed: an older choice (the essentials build) shows "update needed"
  const ff = COMPONENTS.find((c) => c.id === "ffmpeg");
  st.ffmpeg = !(fs.existsSync(L.ffmpeg) && fs.existsSync(L.ffprobe)) ? "missing" : (readStamp(L.stampFf) === ff.file ? "ok" : "update");
  st.mkvtoolnix = fs.existsSync(L.mkvmerge) ? "ok" : "missing";
  const wantV = hashFile(L.reqView), haveV = readStamp(L.stampView);
  st.vsview = !haveV || st.python !== "ok" ? "missing" : (haveV === wantV ? "ok" : "update");
  const list = COMPONENTS.map((c) => ({ id: c.id, name: c.name, what: c.what, size: c.size, required: c.required, state: st[c.id] }));
  const ready = list.every((c) => !c.required || c.state === "ok");
  return { ready, list, folder: L.rt };
}

// ---------------------------------------------------------------- helpers

function download(url, dest, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const part = dest + ".part";
    const out = fs.createWriteStream(part);
    const req = net.request({ url, redirect: "follow" });
    req.setHeader("User-Agent", "DVDRestore/0.1");
    let got = 0, total = 0, last = 0;
    const abort = () => { try { req.abort(); } catch {} out.destroy(); reject(new Error("cancelled")); };
    if (signal) signal.onCancel = abort;
    req.on("response", (res) => {
      if (res.statusCode >= 400) { out.destroy(); return reject(new Error(`download failed: HTTP ${res.statusCode} (${url})`)); }
      total = Number(res.headers["content-length"] || 0);
      res.on("data", (b) => {
        out.write(b); got += b.length;
        const now = Date.now();
        if (now - last > 200) { last = now; onProgress(got, total); }
      });
      res.on("end", () => out.end(() => {
        onProgress(got, total || got);
        if (total && got < total) return reject(new Error("download was cut off - try again"));
        try { fs.renameSync(part, dest); } catch (e) { return reject(e); }
        resolve(dest);
      }));
      res.on("error", (e) => { out.destroy(); reject(e); });
    });
    req.on("error", (e) => { out.destroy(); reject(e); });
    req.end();
  });
}

function run(cmd, args, opts, onLine, signal) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { windowsHide: true, ...opts });
    if (signal) signal.onCancel = () => { try { execFile("taskkill", ["/pid", String(p.pid), "/t", "/f"]); } catch {} };
    let tail = [];
    const feed = (buf) => String(buf).split(/\r?\n|\r/).forEach((l) => {
      if (!l.trim()) return;
      tail.push(l); if (tail.length > 30) tail.shift();
      onLine && onLine(l);
    });
    p.stdout && p.stdout.on("data", feed);
    p.stderr && p.stderr.on("data", feed);
    p.on("error", reject);
    p.on("exit", (code) => code === 0 ? resolve() : reject(new Error(signal && signal.cancelled ? "cancelled" :
      `${path.basename(cmd)} stopped (code ${code}):\n` + tail.slice(-8).join("\n"))));
  });
}

function rmrf(p) { try { fs.rmSync(p, { recursive: true, force: true }); } catch {} }

function freeBytes(dir) {
  try { const s = fs.statfsSync(dir); return s.bavail * s.bsize; } catch { return Infinity; }
}

// ---------------------------------------------------------------- installers

async function get(L, c, ui, signal) {
  fs.mkdirSync(L.downloads, { recursive: true });
  const dest = path.join(L.downloads, c.file);
  if (fs.existsSync(dest)) return dest;                       // left from an earlier try
  ui.status(c.id, "downloading");
  try {
    return await download(c.url, dest, (g, t) => ui.progress(c.id, g, t), signal);
  } catch (e) {
    if (!c.fallback || signal.cancelled) throw e;
    ui.log(c.id, "first address failed, trying " + c.fallback);
    return download(c.fallback, dest, (g, t) => ui.progress(c.id, g, t), signal);
  }
}

async function installPython(L, c, ui, signal) {
  const pkg = await get(L, c, ui, signal);
  ui.status(c.id, "unpacking");
  const dir = path.dirname(L.python);
  rmrf(dir); fs.mkdirSync(dir, { recursive: true });
  await run(TAR, ["-xf", pkg, "-C", dir, "--strip-components=1", "tools"], {}, null, signal);
  if (!fs.existsSync(L.python)) throw new Error("the Python package did not contain python.exe");
  fs.rmSync(pkg, { force: true });
}

function pipEnv(L) {
  const env = { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONNOUSERSITE: "1", PIP_DISABLE_PIP_VERSION_CHECK: "1",
                PIP_NO_WARN_SCRIPT_LOCATION: "1", PIP_CACHE_DIR: path.join(L.downloads, "pip") };
  delete env.PYTHONPATH; delete env.PYTHONHOME; delete env.PIP_USER; delete env.PIP_TARGET; delete env.PIP_PREFIX;
  return env;
}

async function installPackages(L, id, reqFile, stamp, ui, signal) {
  ui.status(id, "downloading");
  // pip prints "Collecting x", "Downloading x (2.9 GB)" and, with --progress-bar raw, "Progress <done> of <total>"
  let cur = "", done = 0, files = 0;
  const onLine = (l) => {
    let m;
    if ((m = l.match(/^Progress (\d+) of (\d+)/))) { ui.progress(id, Number(m[1]), Number(m[2]), cur); return; }
    if ((m = l.match(/^\s*Downloading (\S+)/))) { cur = m[1].split("/").pop(); files++; ui.log(id, l.trim()); return; }
    if ((m = l.match(/^Installing collected packages/))) { ui.status(id, "installing"); ui.progress(id, 0, 0, "unpacking into runtime\\python"); }
    if (/^(Collecting|Building|Preparing|Successfully|ERROR|WARNING)/.test(l.trim())) ui.log(id, l.trim());
  };
  await run(L.python, ["-s", "-m", "pip", "install", "--progress-bar", "raw", "--no-input", "-r", reqFile],
            { cwd: L.root, env: pipEnv(L) }, onLine, signal);
  fs.writeFileSync(stamp, hashFile(reqFile));
}

async function installFfmpeg(L, c, ui, signal) {
  const zip = await get(L, c, ui, signal);
  ui.status(c.id, "unpacking");
  const tmp = path.join(L.downloads, "ffmpeg-unpack");
  rmrf(tmp); fs.mkdirSync(tmp, { recursive: true });
  await run(TAR, ["-xf", zip, "-C", tmp], {}, null, signal);
  const top = fs.readdirSync(tmp).find((d) => fs.existsSync(path.join(tmp, d, "bin", "ffmpeg.exe")));
  if (!top) throw new Error("ffmpeg.exe was not in the download");
  const dst = path.dirname(L.ffmpeg);
  fs.mkdirSync(dst, { recursive: true });
  for (const f of ["ffmpeg.exe", "ffprobe.exe"]) fs.copyFileSync(path.join(tmp, top, "bin", f), path.join(dst, f));
  for (const f of ["LICENSE", "README.txt"]) {
    const s = path.join(tmp, top, f);
    if (fs.existsSync(s)) fs.copyFileSync(s, path.join(dst, f === "LICENSE" ? "LICENSE.txt" : f));
  }
  fs.writeFileSync(L.stampFf, c.file);
  rmrf(tmp); fs.rmSync(zip, { force: true });
}

async function installMkv(L, c, ui, signal) {
  const arc = await get(L, c, ui, signal);
  ui.status(c.id, "unpacking");
  const dst = path.dirname(L.mkvmerge);
  fs.mkdirSync(dst, { recursive: true });
  await run(TAR, ["-xf", arc, "-C", dst, "--strip-components=1", "mkvtoolnix/mkvmerge.exe"], {}, null, signal);
  const tmp = path.join(L.downloads, "mkv-doc");
  fs.mkdirSync(tmp, { recursive: true });
  try {
    await run(TAR, ["-xf", arc, "-C", tmp, "mkvtoolnix/doc/COPYING.txt"], {}, null, signal);
    fs.copyFileSync(path.join(tmp, "mkvtoolnix", "doc", "COPYING.txt"), path.join(dst, "COPYING.txt"));
  } catch {}
  rmrf(tmp);
  if (!fs.existsSync(L.mkvmerge)) throw new Error("mkvmerge.exe was not in the download");
  fs.rmSync(arc, { force: true });
}

// installs every missing / outdated component (+ vsview when asked); ui = {status, progress, log}
async function install(root, { vsview = false } = {}, ui, signal) {
  const L = layout(root);
  const st = Object.fromEntries(check(root).list.map((c) => [c.id, c.state]));
  const todo = COMPONENTS.filter((c) => st[c.id] !== "ok" && (c.required || (c.id === "vsview" && vsview)));
  const need = todo.reduce((a, c) => a + c.size, 0) * 1.6;      // download + unpacked
  fs.mkdirSync(L.rt, { recursive: true });
  if (freeBytes(L.rt) < need)
    throw new Error(`Not enough free space on this drive: about ${(need / 1e9).toFixed(0)} GB needed for the download and install.`);
  for (const c of todo) {
    if (signal.cancelled) throw new Error("cancelled");
    const t0 = Date.now();                                     // a running clock on the row: long steps never look frozen
    const tick = setInterval(() => ui.elapsed(c.id, Math.round((Date.now() - t0) / 1000)), 1000);
    try {
      if (c.id === "python") await installPython(L, c, ui, signal);
      else if (c.id === "packages") await installPackages(L, "packages", L.req, L.stamp, ui, signal);
      else if (c.id === "ffmpeg") await installFfmpeg(L, c, ui, signal);
      else if (c.id === "mkvtoolnix") await installMkv(L, c, ui, signal);
      else if (c.id === "vsview") await installPackages(L, "vsview", L.reqView, L.stampView, ui, signal);
      ui.status(c.id, "ok");
    } catch (e) {
      ui.status(c.id, "failed");
      throw e;
    } finally {
      clearInterval(tick);
    }
  }
  rmrf(path.join(L.downloads, "pip"));                         // the pip cache is only for retries
  try { if (!fs.readdirSync(L.downloads).length) rmrf(L.downloads); } catch {}
}

// the folder layout the app expects; made again if something was deleted (engines/ and temp/ are caches)
function syncLayout(root) {
  const made = [];
  for (const d of ["engines", "temp", "runtime"]) {
    const p = path.join(root, d);
    if (!fs.existsSync(p)) { fs.mkdirSync(p, { recursive: true }); made.push(d); }
  }
  const missing = ["core", "models", "app"].filter((d) => !fs.existsSync(path.join(root, d)));
  return { made, missing };
}

module.exports = { check, install, syncLayout, layout, COMPONENTS };
