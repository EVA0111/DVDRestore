// Safe bridge between the window and the main process.
const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("api", {
  rpc: (method, params) => ipcRenderer.invoke("rpc", method, params),
  onEvent: (cb) => ipcRenderer.on("backend-event", (_e, msg) => cb(msg)),
  openVideo: () => ipcRenderer.invoke("open-video"),
  openFolder: (title) => ipcRenderer.invoke("open-folder", title),
  openLut: () => ipcRenderer.invoke("open-lut"),
  openModel: () => ipcRenderer.invoke("open-model"),
  openMyFilter: () => ipcRenderer.invoke("open-my-filter"),
  openRefs: () => ipcRenderer.invoke("open-refs"),
  showItem: (p) => ipcRenderer.invoke("show-item", p),
  showPath: (p) => ipcRenderer.invoke("show-path", p),
  openGithub: () => ipcRenderer.invoke("open-github"),
  isDir: (p) => ipcRenderer.invoke("is-dir", p),
  saveText: (name, text, filters) => ipcRenderer.invoke("save-text", name, text, filters),
  openText: (filters) => ipcRenderer.invoke("open-text", filters),
  pathForFile: (f) => webUtils.getPathForFile(f),          // drag & drop
  // Dependencies window
  depsCheck: () => ipcRenderer.invoke("deps-check"),
  depsInstall: (opts) => ipcRenderer.invoke("deps-install", opts),
  depsCancel: () => ipcRenderer.invoke("deps-cancel"),
  depsFolder: () => ipcRenderer.invoke("deps-folder"),
  depsOpen: () => ipcRenderer.invoke("deps-open"),
  depsStart: () => ipcRenderer.invoke("deps-start"),
  onDeps: (cb) => ipcRenderer.on("deps-event", (_e, msg) => cb(msg)),
  // updates
  updateCheck: () => ipcRenderer.invoke("update-check"),
  updateDownload: () => ipcRenderer.invoke("update-download"),
  updateCancel: () => ipcRenderer.invoke("update-cancel"),
  updateInstall: () => ipcRenderer.invoke("update-install"),
  openRelease: () => ipcRenderer.invoke("open-release"),
  onUpdate: (cb) => ipcRenderer.on("update-event", (_e, msg) => cb(msg)),
});
