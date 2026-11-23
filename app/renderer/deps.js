// DVD Restore - Dependencies window: shows what the engine needs, installs what is missing into runtime\.
const $ = (id) => document.getElementById(id);
const FIRST = new URLSearchParams(location.search).get("first") === "1";
const LABEL = { ok: "installed", missing: "missing", update: "update needed", downloading: "downloading",
                unpacking: "unpacking", installing: "installing", failed: "failed", waiting: "waiting", optional: "not installed" };
let state = null;
let busy = false;

function gb(n) {
  if (!n) return "";
  return n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : Math.max(1, Math.round(n / 1e6)) + " MB";
}

function render() {
  const list = $("list");
  list.innerHTML = "";
  for (const c of state.list) {
    const el = document.createElement("div");
    el.className = "comp";
    el.id = "c_" + c.id;
    el.innerHTML = `<div class="row"><span class="name"></span><span class="size"></span><span class="el"></span><span class="badge"></span></div>
      <div class="what"></div><div class="bar"><div></div></div><div class="detail"></div>`;
    el.querySelector(".name").textContent = c.name;
    el.querySelector(".size").textContent = c.state === "ok" ? "" : "about " + gb(c.size);
    el.querySelector(".what").textContent = c.what;
    setBadge(el, !c.required && c.state === "missing" ? "optional" : c.state);
    list.appendChild(el);
  }
  const need = state.list.filter((c) => c.required && c.state !== "ok");
  const view = state.list.find((c) => c.id === "vsview");
  $("optView").checked = $("optView").checked || false;
  $("optView").parentElement.style.display = view.state === "ok" ? "none" : "";
  const total = need.reduce((a, c) => a + c.size, 0) + ($("optView").checked && view.state !== "ok" ? view.size : 0);
  $("where").textContent = "Installed into " + state.folder + " - nothing outside this folder is changed.";
  if (state.ready) {
    $("lead").textContent = FIRST ? "Everything is installed." : "Everything DVD Restore needs is installed.";
    $("btnInstall").style.display = view.state === "ok" || !$("optView").checked ? "none" : "";
    $("btnInstall").textContent = "Install vsview";
    $("btnStart").style.display = FIRST ? "" : "none";
  } else {
    $("lead").textContent = (FIRST ? "First start: " : "") +
      `DVD Restore needs ${need.length} more part${need.length > 1 ? "s" : ""} (download about ${gb(total)}). ` +
      "They are installed next to the app, so the folder stays portable.";
    $("btnInstall").style.display = "";
    $("btnInstall").textContent = "Download and install";
    $("btnStart").style.display = "none";
  }
  $("btnInstall").disabled = busy;
}

function setBadge(el, st) {
  const b = el.querySelector(".badge");
  b.textContent = LABEL[st] || st;
  b.className = "badge " + (["downloading", "unpacking", "installing"].includes(st) ? "busy" : st);
  const bar = el.querySelector(".bar");
  bar.style.display = ["downloading", "unpacking", "installing"].includes(st) ? "block" : "none";
  bar.classList.toggle("indet", st !== "downloading");
  if (st === "ok") { el.querySelector(".detail").textContent = ""; el.querySelector(".size").textContent = ""; }
  if (!["downloading", "unpacking", "installing"].includes(st)) el.querySelector(".el").textContent = "";
}

window.api.onDeps((m) => {
  const el = $("c_" + m.id);
  if (!el) return;
  if (m.state) setBadge(el, m.state);
  if (m.total !== undefined) {
    const bar = el.querySelector(".bar");
    if (m.total > 0) {
      bar.classList.remove("indet");
      bar.firstElementChild.style.width = Math.min(100, (100 * m.done) / m.total).toFixed(1) + "%";
      el.querySelector(".detail").textContent = `${m.file ? m.file + "  " : ""}${gb(m.done)} of ${gb(m.total)}`;
    } else {
      bar.classList.add("indet");
      if (m.file) el.querySelector(".detail").textContent = m.file;
    }
  }
  if (m.line) el.querySelector(".detail").textContent = m.line;
  if (m.elapsed !== undefined) {
    const s = m.elapsed;
    el.querySelector(".el").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
});

async function refresh() {
  state = await window.api.depsCheck();
  render();
}

$("optView").addEventListener("change", render);
$("btnFolder").addEventListener("click", () => window.api.depsFolder());
$("btnCancel").addEventListener("click", () => window.api.depsCancel());
$("btnStart").addEventListener("click", () => window.api.depsStart());
$("btnInstall").addEventListener("click", async () => {
  busy = true;
  $("msg").className = "msg"; $("msg").textContent = "";
  $("btnInstall").disabled = true; $("btnCancel").style.display = "";
  for (const c of state.list) {
    if (c.state !== "ok" && (c.required || ($("optView").checked && c.id === "vsview"))) setBadge($("c_" + c.id), "waiting");
  }
  const r = await window.api.depsInstall({ vsview: $("optView").checked });
  busy = false;
  $("btnCancel").style.display = "none";
  if (r.error) {
    $("msg").className = "msg err";
    $("msg").textContent = r.error === "cancelled" ? "Cancelled. Press Download and install to continue - finished parts are kept."
      : r.error + "\n\nPress Download and install to try again: parts that are already finished are kept.";
    const failed = [...document.querySelectorAll(".comp")].filter((el) => el.querySelector(".badge.failed")).map((el) => el.id);
    await refresh();
    failed.forEach((id) => $(id) && setBadge($(id), "failed"));     // keep the red badge on the part that failed
  } else {
    $("msg").className = "msg ok";
    $("msg").textContent = "Done. Everything is installed" + (FIRST ? " - press Start DVD Restore." : ".");
    await refresh();
  }
});

refresh();
