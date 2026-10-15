const totalEl = document.querySelector("#total");
const treeEl = document.querySelector("#tree");
const photosEl = document.querySelector("#photos");
const logEl = document.querySelector("#log");
const browserDialog = document.querySelector("#browser");
const browsePath = document.querySelector("#browse-path");
const browseList = document.querySelector("#browse-list");

async function loadIdentity() {
  try {
    const data = await (await fetch('/api/about')).json();
    document.querySelector('#about-name').textContent = data.name;
    document.querySelector('#about-open').firstChild.textContent = data.name;
    document.title = data.name;
    document.querySelector('#about-author').textContent = data.author;
    document.querySelector('#about-version').textContent = `Version ${data.version}`;
    document.querySelector('#app-version').textContent = data.version;
    const date = new Date(`${data.developed}T12:00:00`);
    document.querySelector('#about-developed').textContent = `Developed: ${date.toLocaleDateString('en-US', {year:'numeric', month:'long', day:'numeric'})}`;
    const link = document.querySelector('#about-github');
    link.href = data.github;
    link.textContent = data.github.replace('https://', '');
  } catch { /* packaged labels stay usable if the server is closing */ }
}
document.querySelector('#about-open').addEventListener('click', async () => {
  await loadIdentity();
  document.querySelector('#about-dialog').showModal();
});
loadIdentity();

const state = {
  tree: null,
  currentImages: [],
  selectedImage: "",
  browsePath: "",
  browseMode: "import",
  job: null,
  jobKind: "",
  plots: null,
  plotKey: "",
  plotsOpenedKey: "",
  browseChecks: new Set(),
  modelPath: "",
  modelClasses: [],
  pendingModelPath: "",
  needsMap: false,
  derivedOutput: "",
  noteCursor: 0,
  configuring: false,
  readyBody: null,
};

let currentFolderButton = null;
let previewEpoch = 0;

function stamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

function emptyLog() {
  logEl.replaceChildren();
  const item = document.createElement("li");
  item.className = "empty";
  const time = document.createElement("time");
  const text = document.createElement("span");
  text.textContent = "Waiting for a job.";
  item.append(time, text);
  logEl.append(item);
}

function logTone(line) {
  if (/fail|error|not found|cannot|stopped/i.test(line)) return "is-error";
  if (/\bmodel\b|\bloaded\b/i.test(line)) return "is-warn";
  if (/\bready\b/i.test(line)) return "is-ready";
  if (/^Imported\b|\bphotos\b|\bWrote\b|\bpreview\b/i.test(line)) return "is-ok";
  return "";
}

function log(line) {
  const empty = logEl.querySelector(".empty");
  if (empty) empty.remove();
  const stick = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 48;
  const item = document.createElement("li");
  const tone = logTone(line);
  if (tone) item.classList.add(tone);
  const time = document.createElement("time");
  time.dateTime = new Date().toISOString();
  time.textContent = stamp();
  const text = document.createElement("span");
  text.textContent = line;
  item.append(time, text);
  logEl.append(item);
  while (logEl.children.length > 400) logEl.firstElementChild.remove();
  if (stick) logEl.scrollTop = logEl.scrollHeight;
}

function showPathEnd() {
  const output = document.querySelector("#output");
  output.title = output.value;
  requestAnimationFrame(() => { output.scrollLeft = output.scrollWidth; });
}

function options() {
  return {
    model: document.querySelector("#method").value,
    batch_size: Number(document.querySelector("#batch-size").value),
    imgsz: Number(document.querySelector("#imgsz").value),
    conf: Number(document.querySelector("#conf").value) / 100,
    qr_mode: document.querySelector("#qr").value,
    save_hops: document.querySelector("#hops").checked,
    output: document.querySelector("#output").value,
    device: document.querySelector("#compute").value || "auto",
    do_gc: document.querySelector("#gamma").checked,
    do_wb: document.querySelector("#wb").checked,
  };
}

function correctionKey() {
  const method = document.querySelector("#method").value;
  const gc = document.querySelector("#gamma").checked ? "gc" : "nogc";
  const wb = document.querySelector("#wb").checked ? "wb" : "nowb";
  const compute = document.querySelector("#compute").value || "auto";
  return `${state.selectedImage}|${method}|${gc}|${wb}|${compute}`;
}

function plotsReady() {
  return Boolean(state.plots && state.plotKey === correctionKey());
}

function syncPlotsButton() {
  const button = document.querySelector("#show-plots");
  if (!button) return;
  button.disabled = Boolean(state.job) || !plotsReady();
}

function setBusy(busy) {
  const hold = busy || state.configuring;
  ["#preview-chart", "#preview-detections", "#preview-corrected", "#run", "#open-browser", "#choose-output", "#load-model", "#compute", "#imgsz", "#edit-classes", "#batch-size", "#conf", "#method", "#gamma", "#wb", "#qr", "#hops", "#output"].forEach((selector) => {
    const element = document.querySelector(selector);
    if (element) element.disabled = selector === "#run" ? hold || state.needsMap : hold;
  });
  document.querySelector("#stop").disabled = !busy;
  syncPlotsButton();
  syncClearButton();
}

function syncClearButton() {
  const button = document.querySelector("#clear-images");
  if (!button) return;
  button.disabled = Boolean(state.job) || !state.tree;
}

function fillCompute(devices, keep) {
  const select = document.querySelector("#compute");
  const current = keep || select.value || "auto";
  select.replaceChildren();
  (devices && devices.length ? devices : [{id: "auto", label: "Auto"}]).forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    if (item.enabled === false) option.disabled = true;
    select.append(option);
  });
  const known = [...select.options].some((option) => option.value === current);
  select.value = known ? current : "auto";
}

let batchApplied = false;

async function refreshVram(setBatch) {
  const select = document.querySelector("#compute");
  const choice = select.value || "auto";
  const response = await fetch(`/api/vram?device=${encodeURIComponent(choice)}`);
  const body = await response.json();
  fillCompute(body.devices, choice);
  document.querySelector("#vram").textContent = body.text || "";
  if (setBatch && body.batch && !batchApplied) {
    document.querySelector("#batch-size").value = body.batch;
    batchApplied = true;
  }
}

async function loadVram() {
  try {
    await refreshVram(true);
  } catch (error) {
    /* the server may still be starting */
  }
}

function absorbNotes(body) {
  (body.lines || []).forEach(log);
  if (Number.isInteger(body.cursor)) state.noteCursor = body.cursor;

}

function showBackendError(text) {
  const status = document.querySelector("#status");
  status.dataset.state = "wait";
  status.textContent = text;
  status.title = text;
}

async function ensureBackend() {
  state.configuring = true;
  setBusy(false);
  try {
    const response = await fetch("/api/backend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        device: document.querySelector("#compute").value || "auto",
        imgsz: Number(document.querySelector("#imgsz").value) || 2048,
        since: state.noteCursor || 0,
      }),
    });
    const body = await response.json();
    absorbNotes(body);
    if (!response.ok) {
      log(body.error || "Could not set the compute device.");
      return false;
    }
    if (body.error) {
      showBackendError(body.error);
      return false;
    }
    state.readyBody=await (await fetch('/api/ready')).json();
    presentStatus(document.querySelector('#status'),state.readyBody);
    return true;
  } catch (error) {
    log("Could not reach the server.");
    return false;
  } finally {
    state.configuring = false;
    setBusy(false);
  }
}

function renderTree(node) {
  const item = document.createElement("li");
  const row = document.createElement("div");
  row.className = "row";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.dataset.path = node.path;
  box.dataset.direct = String(node.images.length);
  box.dataset.files = JSON.stringify(node.images.map((image) => image.path));
  box.addEventListener("change", () => {
    setDescendants(item, box.checked);
    syncAncestors(item);
    updateTotal();
  });
  const name = document.createElement("button");
  name.type = "button";
  name.className = "name";
  name.title = node.name;
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = node.name;
  name.append(label);
  name.addEventListener("click", () => selectFolder(node, name));
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = String(node.total);
  row.append(box, name, count);
  item.append(row);
  if (node.children.length) {
    const children = document.createElement("ul");
    children.className = "children";
    const nested = document.createDocumentFragment();
    node.children.forEach((child) => nested.append(renderTree(child)));
    children.append(nested);
    item.append(children);
  }
  item._node = node;
  return item;
}

function setDescendants(item, checked) {
  item.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.checked = checked;
    box.indeterminate = false;
  });
}

function syncAncestors(item) {
  const parent = item.parentElement?.closest("li");
  if (!parent) return;
  const boxes = [...parent.querySelectorAll(":scope > ul > li > .row input")];
  const all = boxes.every((box) => box.checked && !box.indeterminate);
  const none = boxes.every((box) => !box.checked && !box.indeterminate);
  const parentBox = parent.querySelector(":scope > .row input");
  parentBox.indeterminate = !all && !none;
  parentBox.checked = all;
  syncAncestors(parent);
}

function updateTotal() {
  let discovered = 0;
  let selected = 0;
  document.querySelectorAll("#tree input[type=checkbox]").forEach((box) => {
    const direct = Number(box.dataset.direct);
    discovered += direct;
    if (box.checked || box.indeterminate) selected += direct;
  });
  totalEl.textContent = discovered ? `${selected} selected of ${discovered}` : "No photos";
}

function checkedFolders() {
  const folders = [];
  document.querySelectorAll("#tree input[type=checkbox]").forEach((box) => {
    if ((box.checked || box.indeterminate) && Number(box.dataset.direct) > 0) {
      let files = [];
      try { files = JSON.parse(box.dataset.files || "[]"); } catch (error) { files = []; }
      folders.push({path: box.dataset.path, files});
    }
  });
  return folders;
}

function selectFolder(node, button) {
  if (currentFolderButton && currentFolderButton !== button) currentFolderButton.removeAttribute("aria-current");
  button?.setAttribute("aria-current", "true");
  currentFolderButton = button || null;
  const previous = state.selectedImage;
  state.currentImages = node.images;
  state.selectedImage = "";
  if (previous) {
    clearStage();
    clearPreviewSelection();
    const plotsDialog = document.querySelector("#plots");
    if (plotsDialog?.open) plotsDialog.close();
  }
  syncPlotsButton();
  const note = document.querySelector("#photo-folder");
  const count = node.images.length;
  note.textContent = count ? `${node.name} · ${count}` : node.name;
  note.title = node.name;
  renderPhotos();
}

function clearStage() {
  const empty = document.querySelector("#stage-empty");
  const grid = document.querySelector("#stage-grid");
  grid.replaceChildren();
  grid.hidden = true;
  empty.hidden = false;
  resetZoom();
}

function emptyPhotoNote(text) {
  const item = document.createElement("li");
  item.className = "empty";
  item.textContent = text;
  return item;
}

function renderPhotos() {
  const fragment = document.createDocumentFragment();
  if (!state.tree) {
    fragment.append(emptyPhotoNote("Choose a folder."));
    photosEl.replaceChildren(fragment);
    return;
  }
  if (!state.currentImages.length) {
    fragment.append(emptyPhotoNote("Photos in this folder are inside the folders under it."));
    photosEl.replaceChildren(fragment);
    return;
  }
  state.currentImages.forEach((image) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.path = image.path;
    button.title = image.name;
    button.textContent = image.name;
    if (image.path === state.selectedImage) button.setAttribute("aria-current", "true");
    item.append(button);
    fragment.append(item);
  });
  photosEl.replaceChildren(fragment);
}

function photoButton(event) {
  const button = event.target.closest("button");
  if (!button || !photosEl.contains(button) || !button.dataset.path) return null;
  return button;
}

function selectPhoto(button) {
  const path = button.dataset.path;
  photosEl.querySelectorAll("button[aria-current='true']").forEach((item) => {
    if (item !== button) item.removeAttribute("aria-current");
  });
  button.setAttribute("aria-current", "true");
  state.selectedImage = path;
  syncPlotsButton();
}

function chosenPreviewMode() {
  const button = document.querySelector(".preview-group button[aria-pressed='true']");
  if (!button) return "";
  if (button.id === "preview-chart") return "chart";
  if (button.id === "preview-corrected") return "corrected";
  if (button.id === "preview-detections") return "detections";
  return "";
}

function showOriginalPhoto(path) {
  const name = path.split(/[\\/]/).pop() || "Photo";
  showStage([{url: `/api/photo?path=${encodeURIComponent(path)}`, title: `Original — ${name}`}]);
  const image = document.querySelector("#stage-grid img");
  if (!image) return;
  image.addEventListener("error", () => {
    if (state.selectedImage !== path) return;
    log(`Could not show ${name}.`);
  }, {once: true});
}

function releaseRunningPreview() {
  if (!state.job || state.jobKind !== "preview") return;
  const running = state.job;
  fetch(`/api/jobs/${running}/stop`, {method: "POST"}).catch(() => {});
}

function beginPhoto(path) {
  ++previewEpoch;
  clearPreviewSelection();
  state.plots = null;
  state.plotKey = "";
  state.plotsOpenedKey = "";
  state.plotsFresh = false;
  if (state.jobKind !== "batch") showProgress({done: 0, total: 0, label: ""});
  showOriginalPhoto(path);
  const plotsDialog = document.querySelector("#plots");
  if (plotsDialog?.open) plotsDialog.close();
  releaseRunningPreview();
  void refreshCorrectedCache();
}

function selectImportedImage(imagePath) {
  const folder = imagePath.replace(/[\\/][^\\/]+$/, "");
  const file = imagePath.split(/[\\/]/).pop();
  const box = [...document.querySelectorAll("#tree input")].find((item) => item.dataset.path.toLowerCase() === folder.toLowerCase());
  box?.closest(".row")?.querySelector(".name")?.click();
  const photo = [...document.querySelectorAll("#photos button")].find((item) => item.textContent === file);
  photo?.click();
}

function applySavedFolders() {
  const saved = Number(stored("hopbox.folders"));
  if (saved) applyFolders(saved, true);
  else fitFolders();
}

function showImported(body, focusImage = "") {
  state.tree = body;
  currentFolderButton = null;
  treeEl.replaceChildren(renderTree(body));
  applySavedFolders();
  const derived = `${body.path}_HopBox-Infer`;
  state.derivedOutput = derived;
  document.querySelector("#output").value = derived;
  showPathEnd();
  updateTotal();
  syncClearButton();
  log(`Imported ${body.path} — ${body.total} photos`);
  if (focusImage) {
    selectImportedImage(focusImage);
    return;
  }
  const first = treeEl.querySelector(".name");
  if (first) first.click();
}

function clearImages() {
  if (state.job || !state.tree) return;
  const output = document.querySelector("#output");
  if (state.derivedOutput && output.value === state.derivedOutput) {
    output.value = "";
    showPathEnd();
  }
  state.derivedOutput = "";
  state.tree = null;
  state.currentImages = [];
  state.selectedImage = "";
  state.plots = null;
  state.plotKey = "";
  currentFolderButton = null;
  treeEl.replaceChildren();
  const note = document.querySelector("#photo-folder");
  note.textContent = "";
  note.removeAttribute("title");
  foldersPane.style.flex = "";
  foldersPane.style.maxHeight = "";
  renderPhotos();
  clearStage();
  clearPreviewSelection();
  const plotsDialog = document.querySelector("#plots");
  if (plotsDialog.open) plotsDialog.close();
  updateTotal();
  syncPlotsButton();
  syncClearButton();
  log("Cleared the imported photos.");
}

async function importFolder(path, focusImage = "") {
  const bodyIn = Array.isArray(path) ? {paths: path} : {path};
  const response = await fetch("/api/import", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(bodyIn),
  });
  const body = await response.json();
  if (!response.ok) {
    log(body.error || "Import failed");
    return;
  }
  showImported(body, focusImage);
}

async function openBrowser(path, mode = "import") {
  state.browseMode = mode;
  document.querySelector("#browser h2").textContent = mode === "output" ? "Output folder" : "Import folder";
  document.querySelector("#browse-use").textContent = mode === "output" ? "Save results here" : "Use this folder";
  const response = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
  const body = await response.json();
  if (!response.ok) {
    log(body.error || "Cannot open that folder");
    return;
  }
  state.browsePath = body.path;
  browsePath.value = body.path;
  renderCrumbs(body.path);
  renderDrives(body.drives || []);
  browseList.replaceChildren();
  const entries = [
    ...(body.dirs || []).map((dir) => ({...dir, kind: "folder"})),
    ...(state.browseMode === "output" ? [] : (body.files || []).map((file) => ({...file, kind: "image"}))),
  ];
  if (!entries.length) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = state.browseMode === "output" ? "No folders inside this one." : "No folders or photos in this one.";
    browseList.append(item);
  }
  entries.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "browse-row";
    if (state.browseMode === "import") {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = state.browseChecks.has(entry.path);
      box.addEventListener("change", () => {
        if (box.checked) state.browseChecks.add(entry.path);
        else state.browseChecks.delete(entry.path);
        updateBrowseCount();
      });
      item.append(box);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.kind === "image" ? entry.name : entry.name;
    if (entry.kind === "folder") button.addEventListener("click", () => openBrowser(entry.path, state.browseMode));
    else button.addEventListener("click", () => {
      const box = item.querySelector("input");
      if (!box) return;
      box.checked = !box.checked;
      box.dispatchEvent(new Event("change"));
    });
    item.append(button);
    browseList.append(item);
  });
  updateBrowseCount();
  document.querySelector("#browse-up").onclick = () => body.parent && openBrowser(body.parent, state.browseMode);
  if (!browserDialog.open) browserDialog.showModal();
}

function renderCrumbs(path) {
  const box = document.querySelector("#browse-crumbs");
  box.replaceChildren();
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  let built = "";
  parts.forEach((part, index) => {
    if (index === 0 && /^[A-Za-z]:$/.test(part)) built = `${part}\\`;
    else built = built ? `${built.replace(/[\\/]$/, "")}\\${part}` : part;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = part;
    const target = built;
    button.addEventListener("click", () => openBrowser(target, state.browseMode));
    box.append(button);
  });
}

function renderDrives(drives) {
  const box = document.querySelector("#browse-drives");
  box.replaceChildren();
  drives.forEach((drive) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = drive;
    button.addEventListener("click", () => openBrowser(drive, state.browseMode));
    box.append(button);
  });
}

const zoom = {scale: 1, x: 0, y: 0};

function zoomImage() {
  const image = document.querySelector("#stage-grid img");
  const button = document.querySelector("#zoom-reset");
  if (!image) {
    button.hidden = true;
    return;
  }
  image.style.transform = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`;
  image.classList.toggle("is-zoomed", zoom.scale > 1);
  button.hidden = false;
  button.disabled = zoom.scale === 1 && zoom.x === 0 && zoom.y === 0;
}

function resetZoom() {
  zoom.scale = 1;
  zoom.x = 0;
  zoom.y = 0;
  zoomImage();
}

function showStage(previews) {
  const empty = document.querySelector("#stage-empty");
  const grid = document.querySelector("#stage-grid");
  grid.replaceChildren();
  resetZoom();
  if (!previews?.length) {
    clearStage();
    return;
  }
  previews.forEach((item) => {
    const figure = document.createElement("figure");
    const viewport = document.createElement("div");
    viewport.className = "viewport";
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.title;
    image.draggable = false;
    const caption = document.createElement("figcaption");
    caption.textContent = item.title.replace(/^HopBox /, "");
    viewport.append(image);
    figure.append(viewport, caption);
    grid.append(figure);
  });
  empty.hidden = true;
  grid.hidden = false;
  zoomImage();
}

function showProgress(progress) {
  const meter = document.querySelector("#meter");
  const fill = document.querySelector("#meter-fill");
  const pct = document.querySelector("#meter-pct");
  const label = document.querySelector("#meter-label");
  const count = document.querySelector("#meter-count");
  const done = Number(progress?.done) || 0;
  const total = Number(progress?.total) || 0;
  const ratio = total ? Math.max(0, Math.min(1, done / total)) : 0;
  const text = `${Math.round(ratio * 100)}%`;
  fill.style.width = text;
  fill.setAttribute("aria-valuenow", String(Math.round(ratio * 10000)));
  fill.setAttribute("aria-valuetext", text);
  pct.textContent = text;
  label.textContent = progress && progress.label ? progress.label : (ratio ? "" : "Waiting");
  if (total > 1) {
    count.hidden = false;
    count.textContent = `${Math.min(Math.round(total), Math.round(done))} / ${Math.round(total)}`;
  } else {
    count.hidden = true;
    count.textContent = "";
  }
  meter.classList.toggle("is-active", ratio > 0 && ratio < 1);
  meter.classList.toggle("is-done", ratio >= 1 && total > 0);
}

async function poll(id, plotKey = "", imagePath = "", epoch = 0, mode = "") {
  state.job = id;
  state.jobKind = imagePath ? "preview" : "batch";
  setBusy(true);
  showProgress({done: 0, total: 1, label: "Starting"});
  let seen = 0;
  let inFlight = false;
  const timer = setInterval(async () => {
    if (inFlight) return;
    inFlight = true;
    try {
    let response;
    try {
      response = await fetch(`/api/jobs/${id}`);
    } catch (error) {
      return;
    }
    let job;
    try {
      job = await response.json();
    } catch (error) {
      return;
    }
    if (!response.ok || !Array.isArray(job.log)) {
      clearInterval(timer);
      if (state.job === id) {
        state.job = null;
        state.jobKind = "";
        setBusy(false);
        log(job.error || "Lost the job status.");
      }
      return;
    }
    if (state.job !== id) {
      clearInterval(timer);
      return;
    }
    const stale = (imagePath && state.selectedImage !== imagePath) || (epoch && epoch !== previewEpoch) || (mode && selectedPreview() !== mode);
    if (stale) {
      clearInterval(timer);
      if (state.job === id) {
        state.job = null;
        state.jobKind = "";
        setBusy(false);
      }
      return;
    }
    showProgress(job.progress);
    job.log.slice(seen).forEach(log);
    seen = job.log.length;
    if (job.status === "queued" || job.status === "running") return;
    clearInterval(timer);
    state.job = null;
    state.jobKind = "";
    if (job.status === 'stopped') {
      showProgress({...job.progress, label: 'Stopped'});
      log('Stopped. Run folders again to process the complete selection.');
    } else if (job.status === 'error') {
      showProgress({...job.progress, label: 'Run failed'});
    }
    if (job.plots && job.plots.image === state.selectedImage) {
      state.plots = job.plots;
      state.plotKey = plotKey || correctionKey();
      state.plotsFresh = true;
      state.plotsOpenedKey = "";
    }
    setBusy(false);
    if (job.previews?.length) showStage(job.previews);
    } finally { inFlight = false; }
  }, 250);
}

function markPreview(mode) {
  const ids = {
    chart: "preview-chart",
    corrected: "preview-corrected",
    plots: "show-plots",
    detections: "preview-detections",
  };
  const chosen = ids[mode];
  if (!chosen) return;
  document.querySelectorAll(".preview-group button").forEach((button) => {
    const on = button.id === chosen;
    button.classList.toggle("primary", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

async function startPreview(mode, opts = {}) {
  const epoch = opts.epoch || 0;
  const image = opts.image || state.selectedImage;
  markPreview(mode);
  if (!image) {
    log("Select a photo in the list first.");
    return;
  }
  if (epoch && epoch !== previewEpoch) return;
  if (!(await ensureBackend())) {
    dismissPreview(mode, image);
    return;
  }
  if ((epoch && epoch !== previewEpoch) || state.selectedImage !== image) return;
  const name = image.split(/[\\/]/).pop();
  const labels = {chart: "Chart preview", corrected: "Corrected preview", detections: "Detection preview"};
  log(`${labels[mode] || "Preview"} for ${name}`);
  const plotKey = mode === "corrected" ? correctionKey() : "";
  const deadline = Date.now() + (opts.retry ? 30000 : 0);
  while (true) {
    if ((epoch && epoch !== previewEpoch) || state.selectedImage !== image) return;
    let response;
    try {
      response = await fetch("/api/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...options(), image, mode}),
      });
    } catch (error) {
      log("Could not reach the server.");
      dismissPreview(mode, image);
      return;
    }
    let body;
    try {
      body = await response.json();
    } catch (error) {
      log("Preview failed");
      dismissPreview(mode, image);
      return;
    }
    if (response.ok) {
      if ((epoch && epoch !== previewEpoch) || state.selectedImage !== image) {
        fetch(`/api/jobs/${body.id}/stop`, {method: "POST"}).catch(() => {});
        return;
      }
      poll(body.id, plotKey, image, epoch, mode);
      return;
    }
    const busy = response.status === 409 && /already running/i.test(body.error || "");
    if (opts.retry && busy && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 250));
      continue;
    }
    log(body.error || "Preview failed");
    dismissPreview(mode, image);
    return;
  }
}

async function startRun() {
  const folders = checkedFolders();
  if (!folders.length) {
    log("Check at least one folder that contains photos.");
    return;
  }
  if (!document.querySelector("#output").value.trim()) {
    log("Choose an output folder first.");
    return;
  }
  if (!(await ensureBackend())) return;
  log(`Running ${folders.length} folder(s) with ${document.querySelector("#method").selectedOptions[0].textContent}`);
  try {
  const response = await fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({...options(), folders}),
  });
  const body = await response.json();
  if (!response.ok) {
    log(body.error || "Run failed");
    return;
  }
  poll(body.id);
  } catch {
    setBusy(false);
    log('Could not reach HopBox Infer. Check that the app is still running.');
  }
}

async function choosePath(purpose) {
  const start = purpose === "output" ? document.querySelector("#output").value : (state.browsePath || "");
  let response;
  try {
    response = await fetch("/api/pick", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({purpose, start}),
    });
  } catch (error) {
    log("Could not open the folder window.");
    return null;
  }
  const body = await response.json();
  if (!response.ok) {
    log(body.error || "Could not open the folder window.");
    return null;
  }
  if (body.cancelled) return null;
  return body;
}

document.querySelector("#clear-images").addEventListener("click", clearImages);
photosEl.addEventListener("click", (event) => {
  const button = photoButton(event);
  if (!button) return;
  const path = button.dataset.path;
  const changed = state.selectedImage !== path;
  selectPhoto(button);
  if (!changed) return;
  beginPhoto(path);
});
photosEl.addEventListener("dblclick", (event) => {
  const button = photoButton(event);
  if (!button) return;
  const path = button.dataset.path;
  const changed = state.selectedImage !== path;
  selectPhoto(button);
  if (!changed) return;
  beginPhoto(path);
});
document.querySelector("#open-browser").addEventListener("click", async (event) => {
  if (event.shiftKey) {
    await openBrowser(state.browsePath || "", "import");
    return;
  }
  const button = document.querySelector("#open-browser");
  button.disabled = true;
  try {
    const picked = await choosePath("import");
    if (!picked) return;
    const paths = picked.paths || (picked.path ? [picked.path] : []);
    if (paths.length === 1 && picked.kind === "image") await importFolder(picked.folder, picked.path);
    else await importFolder(paths);
  } finally {
    if (!document.querySelector("#run").disabled) button.disabled = false;
  }
});
document.querySelector("#choose-output").addEventListener("click", async () => {
  const button = document.querySelector("#choose-output");
  button.disabled = true;
  try {
    const picked = await choosePath("output");
    if (!picked || !picked.path) return;
    document.querySelector("#output").value = picked.path;
    showPathEnd();
    log(`Output folder set to ${picked.path}`);
  } finally {
    if (!document.querySelector("#run").disabled) button.disabled = false;
  }
});
function updateBrowseCount() {
  const button = document.querySelector("#browse-use");
  if (state.browseMode === "output") {
    button.textContent = "Use this folder";
    return;
  }
  const count = state.browseChecks.size;
  button.textContent = count ? `Import ${count} selected` : "Use this folder";
}

document.querySelector("#browse-use").addEventListener("click", async () => {
  browserDialog.close();
  if (state.browseMode === "output") {
    document.querySelector("#output").value = state.browsePath;
    showPathEnd();
    log(`Output folder set to ${state.browsePath}`);
    return;
  }
  const selected = [...state.browseChecks];
  state.browseChecks = new Set();
  await importFolder(selected.length ? selected : state.browsePath);
});
function selectedPreview() {
  const button = document.querySelector(".preview-group button[aria-pressed='true']");
  if (!button) return "";
  if (button.id === "preview-chart") return "chart";
  if (button.id === "preview-corrected") return "corrected";
  if (button.id === "preview-detections") return "detections";
  if (button.id === "show-plots") return "plots";
  return "";
}

function clearPreviewSelection() {
  document.querySelectorAll(".preview-group button").forEach((button) => {
    button.classList.remove("primary");
    button.setAttribute("aria-pressed", "false");
  });
}

function viewLabel(mode) {
  return {chart: "Chart", corrected: "Corrected", detections: "Detections", plots: "correction plots"}[mode] || "Preview";
}

function dismissPreview(mode, image) {
  if (selectedPreview() !== mode) return;
  clearPreviewSelection();
  const dialog = document.querySelector("#plots");
  if (dialog?.open) dialog.close();
  if (image && state.selectedImage === image) showOriginalPhoto(image);
}

function previewSettingAffects(mode, kind) {
  if (!mode) return false;
  if (kind === "detect") return mode === "detections";
  if (kind === "compute") return true;
  return mode === "corrected" || mode === "plots" || mode === "detections";
}

function onPreviewSetting(kind) {
  const mode = selectedPreview();
  if (previewSettingAffects(mode, kind)) {
    clearPreviewSelection();
    const dialog = document.querySelector("#plots");
    if (dialog?.open) dialog.close();
    if (state.selectedImage) showOriginalPhoto(state.selectedImage);
  }
  if (kind === "color" || kind === "compute") {
    state.plots = null;
    state.plotKey = "";
    state.plotsOpenedKey = "";
    state.plotsFresh = false;
    void refreshCorrectedCache();
  }
  syncPlotsButton();
}

let cacheCheck = 0;

async function lookupCache(mode) {
  const image = state.selectedImage;
  if (!image) return null;
  let response;
  try {
    response = await fetch("/api/preview-cache", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...options(), image, mode}),
    });
  } catch (error) {
    return null;
  }
  if (!response.ok) return null;
  try {
    const body = await response.json();
    return body.hit ? body : null;
  } catch (error) {
    return null;
  }
}

async function refreshCorrectedCache() {
  const token = ++cacheCheck;
  const image = state.selectedImage;
  if (!image) {
    syncPlotsButton();
    return;
  }
  const hit = await lookupCache("corrected");
  if (token !== cacheCheck || state.selectedImage !== image) return;
  if (hit?.plots) {
    state.plots = hit.plots;
    state.plotKey = correctionKey();
  }
  syncPlotsButton();
}

async function activatePreview(mode) {
  if (!state.selectedImage) {
    log("Select a photo in the list first.");
    return;
  }
  if (selectedPreview() === mode) {
    clearPreviewSelection();
    const dialog = document.querySelector("#plots");
    if (dialog?.open) dialog.close();
    showOriginalPhoto(state.selectedImage);
    return;
  }
  if (mode === "plots") {
    await openPlots();
    return;
  }
  const image = state.selectedImage;
  const epoch = previewEpoch;
  markPreview(mode);
  const hit = await lookupCache(mode);
  if (epoch !== previewEpoch || state.selectedImage !== image || selectedPreview() !== mode) return;
  if (hit?.previews?.length) {
    const name = image.split(/[\\/]/).pop();
    log(`${name}: reused ${viewLabel(mode)}.`);
    if (mode === "corrected" && hit.plots) {
      state.plots = hit.plots;
      state.plotKey = correctionKey();
      syncPlotsButton();
    }
    showStage(hit.previews);
    return;
  }
  await startPreview(mode, {epoch, image});
}

async function openPlots() {
  const image = state.selectedImage;
  const epoch = previewEpoch;
  if (!image) return;
  markPreview("plots");
  let plots = plotsReady() ? state.plots : null;
  let reused = false;
  if (state.plotsFresh && plots) {
    state.plotsFresh = false;
  } else if (!plots) {
    const hit = await lookupCache("plots");
    if (epoch !== previewEpoch || state.selectedImage !== image || selectedPreview() !== "plots") return;
    if (hit?.plots) {
      plots = hit.plots;
      state.plots = plots;
      state.plotKey = correctionKey();
      reused = true;
    }
  } else {
    reused = true;
  }
  if (!plots) {
    clearPreviewSelection();
    syncPlotsButton();
    return;
  }
  if (reused) log(`${image.split(/[\\/]/).pop()}: reused correction plots.`);
  state.plotsOpenedKey = correctionKey();
  syncPlotsButton();
  const dialog = document.querySelector("#plots");
  if (!dialog.open) dialog.showModal();
  drawPlots(plots.patches || []);
}

document.querySelector("#preview-chart").addEventListener("click", () => activatePreview("chart"));
document.querySelector("#preview-corrected").addEventListener("click", () => activatePreview("corrected"));
document.querySelector("#preview-detections").addEventListener("click", () => activatePreview("detections"));
document.querySelector("#method").addEventListener("change", () => onPreviewSetting("color"));
document.querySelector("#gamma").addEventListener("change", () => onPreviewSetting("color"));
document.querySelector("#wb").addEventListener("change", () => onPreviewSetting("color"));
document.querySelector("#compute").addEventListener("change", async () => {
  onPreviewSetting("compute");
  try { await refreshVram(false); } catch (error) { /* keep the previous plan text */ }
  await ensureBackend();
});
let imgszTimer = 0;
document.querySelector("#imgsz").addEventListener("change", () => {
  onPreviewSetting("detect");
  clearTimeout(imgszTimer);
  imgszTimer = setTimeout(() => { ensureBackend(); }, 350);
});
document.querySelector("#conf").addEventListener("change", () => onPreviewSetting("detect"));
document.querySelector("#show-plots").addEventListener("click", () => activatePreview("plots"));
document.querySelector("#plots-close").addEventListener("click", () => document.querySelector("#plots").close());
document.querySelector("#plots").addEventListener("close", () => {
  if (selectedPreview() !== "plots") return;
  clearPreviewSelection();
  if (state.selectedImage) showOriginalPhoto(state.selectedImage);
});
document.querySelector("#plots-export").addEventListener("click", exportPlots);
document.querySelector("#load-model").addEventListener("click", loadDetectionModel);
document.querySelector("#edit-classes").addEventListener("click", async () => {
  try {
    const body = await (await fetch("/api/model")).json();
    applyModelState(body);
    openClassMap(body);
  } catch (error) {
    openClassMap({path: state.modelPath, classes: state.modelClasses, count: state.modelClasses.length});
  }
});
document.querySelector("#class-map-save").addEventListener("click", saveClassMap);
document.querySelector("#clear-log").addEventListener("click", emptyLog);
document.querySelector("#browse-go").addEventListener("click", () => openBrowser(browsePath.value, state.browseMode));
document.querySelector("#browse-form").addEventListener("submit", (event) => {
  if (document.activeElement === browsePath) {
    event.preventDefault();
    openBrowser(browsePath.value, state.browseMode);
  }
});
browsePath.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  openBrowser(browsePath.value, state.browseMode);
});
document.querySelector("#output").addEventListener("blur", showPathEnd);
document.querySelector("#run").addEventListener("click", startRun);
document.querySelector("#stop").addEventListener("click", () => {
  if (state.job) fetch(`/api/jobs/${state.job}/stop`, {method: "POST"});
});

const stageWell = document.querySelector("#stage-well");
let pan = null;
stageWell.addEventListener("wheel", (event) => {
  const viewport = document.querySelector("#stage-grid .viewport");
  if (!viewport) return;
  event.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const cx = event.clientX - (rect.left + rect.width / 2);
  const cy = event.clientY - (rect.top + rect.height / 2);
  const next = Math.min(12, Math.max(1, zoom.scale * (event.deltaY < 0 ? 1.1 : 1 / 1.1)));
  const ix = (cx - zoom.x) / zoom.scale;
  const iy = (cy - zoom.y) / zoom.scale;
  zoom.scale = next;
  if (next === 1) {
    zoom.x = 0;
    zoom.y = 0;
  } else {
    zoom.x = cx - ix * next;
    zoom.y = cy - iy * next;
  }
  zoomImage();
}, {passive: false});
stageWell.addEventListener("pointerdown", (event) => {
  if (event.target.closest("#zoom-reset") || zoom.scale <= 1 || !event.target.closest(".viewport")) return;
  pan = {id: event.pointerId, x: event.clientX, y: event.clientY, ox: zoom.x, oy: zoom.y};
  const image = document.querySelector("#stage-grid img");
  image?.classList.add("is-panning");
  stageWell.setPointerCapture(event.pointerId);
});
stageWell.addEventListener("pointermove", (event) => {
  if (!pan || pan.id !== event.pointerId) return;
  zoom.x = pan.ox + (event.clientX - pan.x);
  zoom.y = pan.oy + (event.clientY - pan.y);
  zoomImage();
});
function endPan(event) {
  if (!pan || (event && pan.id !== event.pointerId)) return;
  pan = null;
  document.querySelector("#stage-grid img")?.classList.remove("is-panning");
}
stageWell.addEventListener("pointerup", endPan);
stageWell.addEventListener("pointercancel", endPan);
stageWell.addEventListener("dblclick", (event) => {
  if (event.target.closest(".viewport")) resetZoom();
});
document.querySelector("#zoom-reset").addEventListener("click", resetZoom);

document.querySelector("#exit").addEventListener("click", () => {
  document.querySelector("#exit-dialog").showModal();
});
document.querySelector("#exit-confirm").addEventListener("click", async () => {
  const button = document.querySelector("#exit-confirm");
  button.disabled = true;
  document.querySelector("#exit-dialog").close();
  log("Closing HopBox Infer.");
  document.querySelector("#exit").disabled = true;
  try {
    await fetch("/api/exit", {method: "POST"});
  } catch (error) {
    /* the server may already be gone */
  }
  window.close();
  window.location.replace("/closing");
});

loadVram();

const workspaceEl = document.querySelector(".workspace");
const railEl = document.querySelector("#rail");
const foldersPane = document.querySelector("#folders-pane");
const splitRail = document.querySelector("#split-rail");
const splitLists = document.querySelector("#split-lists");

function stored(key) {
  try { return localStorage.getItem(key); } catch (error) { return null; }
}
function store(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, String(value));
  } catch (error) { /* the layout still works for this session */ }
}
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
function railLimits() {
  const width = workspaceEl.clientWidth || window.innerWidth;
  if (getComputedStyle(splitRail).display === "none") return {min: 200, max: Math.max(220, width)};
  const inspector = document.querySelector(".inspector").getBoundingClientRect();
  const beside = inspector.width > 80 && inspector.left > railEl.getBoundingClientRect().left + 80;
  const reserved = (beside ? inspector.width : 0) + 280;
  const cap = beside ? width * 0.42 : width * 0.46;
  return {min: 200, max: Math.max(220, Math.min(cap, width - reserved))};
}
function folderLimits() {
  const row = document.querySelector(".import-block") || document.querySelector(".import-row") || document.querySelector("#open-browser");
  const available = railEl.clientHeight - row.offsetHeight - splitLists.offsetHeight - 16;
  const min = Math.min(96, Math.max(64, available / 2));
  return {min, max: Math.max(min, available - 140)};
}
function applyRail(px) {
  const limits = railLimits();
  const next = clamp(px, limits.min, limits.max);
  workspaceEl.style.setProperty("--rail", `${Math.round(next)}px`);
  splitRail.setAttribute("aria-valuenow", String(Math.round(next)));
  splitRail.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitRail.setAttribute("aria-valuemax", String(Math.round(limits.max)));
  return next;
}
function fitFolders() {
  if (stored("hopbox.folders")) return;
  requestAnimationFrame(() => {
    if (stored("hopbox.folders")) return;
    const head = foldersPane.querySelector(".pane-head").offsetHeight;
    const content = [...treeEl.children].reduce((sum, item) => sum + item.offsetHeight, 0);
    if (content < 80) return;
    applyFolders(head + content + 8, false);
  });
}
function applyFolders(px, lock) {
  const limits = folderLimits();
  const next = clamp(px, limits.min, limits.max);
  foldersPane.style.flex = `0 1 ${Math.round(next)}px`;
  foldersPane.style.maxHeight = lock ? "none" : "";
  splitLists.setAttribute("aria-valuenow", String(Math.round(next)));
  splitLists.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitLists.setAttribute("aria-valuemax", String(Math.round(limits.max)));
  return next;
}
function bindDrag(handle, {axis, read, write, reset, sign = 1}) {
  const interactive = (event) => event.target.closest("button, a, input, select, textarea");
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || interactive(event)) return;
    event.preventDefault();
    try { handle.setPointerCapture(event.pointerId); } catch (error) { /* untrusted events still resize */ }
    const start = axis === "x" ? event.clientX : event.clientY;
    const origin = read();
    document.body.classList.add("is-dragging");
    document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";
    const move = (ev) => {
      const delta = ((axis === "x" ? ev.clientX : ev.clientY) - start) * sign;
      write(origin + delta, false);
    };
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      document.body.classList.remove("is-dragging");
      document.body.style.cursor = "";
      write(read(), true);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  });
  handle.addEventListener("keydown", (event) => {
    if (interactive(event)) return;
    const step = event.shiftKey ? 48 : 16;
    const current = read();
    if (event.key === "Home") {
      reset();
      event.preventDefault();
      return;
    }
    const delta = axis === "x"
      ? (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0)
      : (event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0);
    if (!delta) return;
    write(current + delta * sign, true);
    event.preventDefault();
  });
  handle.addEventListener("dblclick", (event) => {
    if (interactive(event)) return;
    reset();
  });
}

bindDrag(splitRail, {
  axis: "x",
  read: () => railEl.getBoundingClientRect().width,
  write: (px, persist) => {
    const next = applyRail(px);
    if (persist) store("hopbox.rail", Math.round(next));
  },
  reset: () => {
    workspaceEl.style.removeProperty("--rail");
    store("hopbox.rail", null);
    splitRail.removeAttribute("aria-valuenow");
  },
});
bindDrag(splitLists, {
  axis: "y",
  read: () => foldersPane.getBoundingClientRect().height,
  write: (px, persist) => {
    const next = applyFolders(px, true);
    if (persist) store("hopbox.folders", Math.round(next));
  },
  reset: () => {
    foldersPane.style.flex = "";
    foldersPane.style.maxHeight = "";
    store("hopbox.folders", null);
    splitLists.removeAttribute("aria-valuenow");
  },
});

const inspectorEl = document.querySelector(".inspector");
const splitInspector = document.querySelector("#split-inspector");

function inspectorActive() {
  return getComputedStyle(splitInspector).display !== "none";
}
function inspectorLimits() {
  const min = 280;
  const hardMax = 640;
  const width = workspaceEl.clientWidth || window.innerWidth;
  const rail = railEl.getBoundingClientRect().width;
  const splits = splitRail.offsetWidth + splitInspector.offsetWidth;
  const available = width - rail - splits - 220;
  return {min, max: Math.max(min, Math.min(hardMax, available))};
}
function inspectorChrome() {
  const style = getComputedStyle(inspectorEl);
  return (parseFloat(style.marginLeft) || 0) + (parseFloat(style.marginRight) || 0);
}
function applyInspector(px) {
  const limits = inspectorLimits();
  const next = clamp(px, limits.min, limits.max);
  workspaceEl.style.setProperty("--inspector", `${Math.round(next + inspectorChrome())}px`);
  splitInspector.setAttribute("aria-valuenow", String(Math.round(next)));
  splitInspector.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitInspector.setAttribute("aria-valuemax", String(Math.round(limits.max)));
  return next;
}
function publishInspectorAria() {
  const limits = inspectorLimits();
  splitInspector.setAttribute("aria-valuenow", String(Math.round(inspectorEl.getBoundingClientRect().width)));
  splitInspector.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitInspector.setAttribute("aria-valuemax", String(Math.round(limits.max)));
}
bindDrag(splitInspector, {
  axis: "x",
  sign: -1,
  read: () => inspectorEl.getBoundingClientRect().width,
  write: (px, persist) => {
    const next = applyInspector(px);
    if (persist) store("hopbox.inspector", Math.round(next));
  },
  reset: () => {
    workspaceEl.style.removeProperty("--inspector");
    store("hopbox.inspector", null);
    publishInspectorAria();
  },
});

const footEl = document.querySelector(".foot");
const splitLog = document.querySelector("#split-log");
const logHead = document.querySelector(".log-head");

function logLimits() {
  const top = document.querySelector(".top").offsetHeight;
  const meter = document.querySelector(".meter").offsetHeight;
  const head = logHead.offsetHeight;
  const grip = splitLog.offsetHeight;
  const reserved = top + meter + head + grip + 260;
  const min = 72;
  const max = Math.max(min, Math.min(420, window.innerHeight - reserved));
  return {min, max};
}
function applyLog(px) {
  const limits = logLimits();
  const next = clamp(px, limits.min, limits.max);
  footEl.style.setProperty("--log", `${Math.round(next)}px`);
  splitLog.setAttribute("aria-valuenow", String(Math.round(next)));
  splitLog.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitLog.setAttribute("aria-valuemax", String(Math.round(limits.max)));
  return next;
}
function publishLogAria() {
  const limits = logLimits();
  splitLog.setAttribute("aria-valuenow", String(Math.round(logEl.getBoundingClientRect().height)));
  splitLog.setAttribute("aria-valuemin", String(Math.round(limits.min)));
  splitLog.setAttribute("aria-valuemax", String(Math.round(limits.max)));
}
const logDrag = {
  axis: "y",
  sign: -1,
  read: () => logEl.getBoundingClientRect().height,
  write: (px, persist) => {
    const next = applyLog(px);
    if (persist) store("hopbox.log", Math.round(next));
  },
  reset: () => {
    footEl.style.removeProperty("--log");
    store("hopbox.log", null);
    publishLogAria();
  },
};
bindDrag(splitLog, logDrag);
bindDrag(logHead, logDrag);

const savedRail = Number(stored("hopbox.rail"));
if (savedRail) applyRail(savedRail);
const savedFolders = Number(stored("hopbox.folders"));
if (savedFolders && state.tree) applyFolders(savedFolders);
const savedLog = Number(stored("hopbox.log"));
if (savedLog) applyLog(savedLog);
else publishLogAria();
if (inspectorActive()) {
  const savedInspector = Number(stored("hopbox.inspector"));
  if (savedInspector) applyInspector(savedInspector);
  else publishInspectorAria();
}
window.addEventListener("resize", () => {
  const rail = Number(stored("hopbox.rail"));
  if (rail) applyRail(rail);
  const logHeight = Number(stored("hopbox.log"));
  if (logHeight) applyLog(logHeight);
  else publishLogAria();
  if (inspectorActive()) {
    const inspectorWidth = Number(stored("hopbox.inspector"));
    if (inspectorWidth) applyInspector(inspectorWidth);
    else publishInspectorAria();
  }
  if (!state.tree) return;
  const folders = Number(stored("hopbox.folders"));
  if (folders) applyFolders(folders, true);
  else if (foldersPane.style.flex) fitFolders();
});
syncClearButton();

function presentStatus(status, body) {
  const raw = body.text || "";
  status.dataset.state = body.ok && raw.startsWith("Models ready") ? "ready" : "wait";
  if (!raw.startsWith("Models ready")) {
    status.textContent = raw;
    status.title = "";
    return;
  }
  const detail = body.detail || raw.replace(/^Models ready\.\s*/, "");
  let chip = /^CPU\b/.test(detail) ? "CPU" : "GPU";
  status.textContent = `Models ready · ${chip}`;
  status.title = detail;
}

function meanRgb(rgb) {
  if (!rgb || rgb.length < 3) return null;
  return (Number(rgb[0]) + Number(rgb[1]) + Number(rgb[2])) / 3;
}

function paintColor(rgb, alpha = 1) {
  const channel = (value) => Math.round(Math.max(0, Math.min(1, Number(value))) * 255);
  return `rgba(${channel(rgb[0])}, ${channel(rgb[1])}, ${channel(rgb[2])}, ${alpha})`;
}

function fitLine(points) {
  const count = points.length;
  if (count < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  points.forEach(([x, y]) => { sx += x; sy += y; sxx += x * x; sxy += x * y; });
  const denom = count * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return null;
  const slope = (count * sxy - sx * sy) / denom;
  return {slope, intercept: (sy - slope * sx) / count};
}

function canvasContext(canvas, width, height) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return context;
}

function drawScatter(canvas, patches) {
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 320;
  const context = canvasContext(canvas, width, height);
  context.fillStyle = "#fffcf8";
  context.fillRect(0, 0, width, height);
  const pad = {l: 58, r: 16, t: 28, b: 42};
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const sx = (value) => pad.l + value * plotW;
  const sy = (value) => pad.t + (1 - value) * plotH;
  context.strokeStyle = "#e4dfd6";
  context.lineWidth = 1;
  context.fillStyle = "#6d6860";
  context.font = "11px Bahnschrift, Segoe UI, sans-serif";
  context.textAlign = "center";
  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    context.beginPath();
    context.moveTo(sx(0), sy(value));
    context.lineTo(sx(1), sy(value));
    context.moveTo(sx(value), sy(0));
    context.lineTo(sx(value), sy(1));
    context.stroke();
    context.fillText(value.toFixed(1), sx(value), height - 18);
    context.textAlign = "right";
    context.fillText(value.toFixed(1), pad.l - 8, sy(value) + 4);
    context.textAlign = "center";
  }
  context.strokeStyle = "#1c1c1a";
  context.strokeRect(pad.l, pad.t, plotW, plotH);
  context.fillText("Reference mean RGB (sRGB 0–1)", pad.l + plotW / 2, height - 4);
  context.save();
  context.translate(14, pad.t + plotH / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Measured mean RGB (sRGB 0–1)", 0, 0);
  context.restore();
  context.save();
  context.beginPath();
  context.rect(pad.l, pad.t, plotW, plotH);
  context.clip();
  context.setLineDash([6, 4]);
  context.strokeStyle = "rgba(109,104,96,0.7)";
  context.beginPath();
  context.moveTo(sx(0), sy(0));
  context.lineTo(sx(1), sy(1));
  context.stroke();
  const original = [];
  const corrected = [];
  patches.forEach((patch) => {
    const x = meanRgb(patch.reference);
    const y = meanRgb(patch.measured);
    if (x == null || y == null) return;
    original.push([x, y]);
    context.fillStyle = paintColor(patch.measured, 0.75);
    context.beginPath();
    context.moveTo(sx(x), sy(y) - 6);
    context.lineTo(sx(x) + 5.5, sy(y) + 4);
    context.lineTo(sx(x) - 5.5, sy(y) + 4);
    context.closePath();
    context.fill();
    const cy = meanRgb(patch.corrected);
    if (cy == null) return;
    corrected.push([x, cy]);
    context.beginPath();
    context.fillStyle = paintColor(patch.corrected, 0.9);
    context.strokeStyle = "#1c1c1a";
    context.lineWidth = 1;
    context.setLineDash([]);
    context.arc(sx(x), sy(cy), 4.2, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
  [[original, "#f59e0b", [5, 4]], [corrected, "#10b981", []]].forEach(([points, color, dash]) => {
    const line = fitLine(points);
    if (!line) return;
    context.beginPath();
    context.setLineDash(dash);
    context.strokeStyle = color;
    context.lineWidth = 1.4;
    context.moveTo(sx(0), sy(line.intercept));
    context.lineTo(sx(1), sy(line.slope + line.intercept));
    context.stroke();
  });
  context.restore();
}

function drawBoard(canvas, patches, key) {
  const width = canvas.clientWidth || 320;
  const height = canvas.clientHeight || 280;
  const context = canvasContext(canvas, width, height);
  context.fillStyle = "#1c1e21";
  context.fillRect(0, 0, width, height);
  const count = patches.length || 1;
  const cols = count === 24 ? 6 : Math.max(1, Math.ceil(Math.sqrt(count)));
  const rows = Math.max(1, Math.ceil(count / cols));
  const inset = 10;
  const gap = Math.max(4, Math.round(Math.min(width, height) * 0.03));
  const square = Math.max(2, Math.min(
    (width - inset * 2 - gap * (cols - 1)) / cols,
    (height - inset * 2 - gap * (rows - 1)) / rows,
  ));
  const gridW = cols * square + gap * (cols - 1);
  const gridH = rows * square + gap * (rows - 1);
  const ox = (width - gridW) / 2;
  const oy = (height - gridH) / 2;
  patches.forEach((patch, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x = ox + col * (square + gap);
    const y = oy + row * (square + gap);
    if (patch.reference?.length >= 3) {
      context.fillStyle = paintColor(patch.reference);
      context.fillRect(x, y, square, square);
    }
    const inner = patch[key];
    if (inner?.length >= 3) {
      const side = square * 0.55;
      context.fillStyle = paintColor(inner);
      context.fillRect(x + (square - side) / 2, y + (square - side) / 2, side, side);
    }
  });
}

function drawPlots(patches) {
  drawScatter(document.querySelector("#plot-scatter"), patches);
  drawBoard(document.querySelector("#plot-original"), patches, "measured");
  drawBoard(document.querySelector("#plot-corrected"), patches, "corrected");
  document.querySelector("#plot-count").textContent = `${patches.length} patches`;
}

function exportPlots() {
  const patches = state.plots?.patches || [];
  const width = 1100;
  const height = 760;
  const canvas = document.createElement("canvas");
  const ratio = 2;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.fillStyle = "#f7f6f3";
  context.fillRect(0, 0, width, height);
  const scatter = document.querySelector("#plot-scatter");
  const original = document.querySelector("#plot-original");
  const corrected = document.querySelector("#plot-corrected");
  context.drawImage(scatter, 24, 24, width - 48, 340);
  context.drawImage(original, 24, 384, (width - 72) / 2, 340);
  context.drawImage(corrected, 48 + (width - 72) / 2, 384, (width - 72) / 2, 340);
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = "hopbox-correction-plots.png";
  link.click();
}

function classCountLabel(count) {
  const n = Number(count) || 0;
  return `${n} ${n === 1 ? "class" : "classes"}`;
}

function applyModelState(body) {
  const modelChanged=body.path && state.modelPath && body.path !== state.modelPath;
  if (modelChanged && state.selectedImage) {
    clearPreviewSelection();
    showOriginalPhoto(state.selectedImage);
  }
  state.modelPath = body.path || "";
  state.modelClasses = body.classes || [];
  state.needsMap = Boolean(body.needs_map);
  const name = document.querySelector("#model-name");
  name.textContent = body.name || "hops_fp32.onnx";
  name.title = body.path || "";
  const note = document.querySelector("#model-note");
  if (note) {
    if (state.needsMap) {
      const count = Number.isInteger(body.count) ? body.count : state.modelClasses.length;
      note.hidden = false;
      note.textContent = `${classCountLabel(count)} ${count === 1 ? "needs" : "need"} a role before a run.`;
    } else {
      note.hidden = true;
      note.textContent = "";
    }
  }
  const run = document.querySelector("#run");
  if (run && !state.job) run.disabled = state.needsMap;
}

function refreshClassWarn() {
  const roles = [...document.querySelectorAll("#class-map-body select")].map((item) => item.value);
  document.querySelector("#class-map-warn").hidden = roles.includes("measure");
}

function keepSingleRoles() {
  ["qr", "card"].forEach((role) => {
    let seen = false;
    document.querySelectorAll("#class-map-body select").forEach((select) => {
      if (select.value !== role) return;
      if (seen) select.value = "skip";
      else seen = true;
    });
  });
}

function openClassMap(body) {
  state.pendingModelPath = body.path || state.modelPath;
  const classes = [...(body.classes || [])];
  if (classes.some((row) => Number.isInteger(row.index))) {
    classes.sort((a, b) => (Number.isInteger(a.index) ? a.index : 1e9) - (Number.isInteger(b.index) ? b.index : 1e9));
  }
  const count = Number.isInteger(body.count) ? body.count : classes.length;
  document.querySelector("#class-map-count").textContent = `This model has ${classCountLabel(count)}.`;
  const tbody = document.querySelector("#class-map-body");
  tbody.replaceChildren();
  classes.forEach((row, index) => {
    const tr = document.createElement("tr");
    const modelIndex = Number.isInteger(row.index) ? row.index : index;
    tr.dataset.source = row.source;
    tr.dataset.index = String(modelIndex);
    const indexCell = document.createElement("td");
    indexCell.className = "class-index";
    indexCell.textContent = String(modelIndex);
    const from = document.createElement("td");
    from.textContent = row.source;
    const nameCell = document.createElement("td");
    const input = document.createElement("input");
    input.type = "text";
    input.value = row.display || row.source;
    input.setAttribute("aria-label", `Display name for ${row.source}`);
    nameCell.append(input);
    const roleCell = document.createElement("td");
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Role for ${row.source}`);
    [["measure", "Measure (extract features)"], ["qr", "QR code (naming/QR)"], ["card", "Color card"], ["skip", "Skip"]].forEach(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      select.append(option);
    });
    if (row.role === "draw") {
      const option = document.createElement("option");
      option.value = "draw";
      option.textContent = "Draw";
      select.append(option);
    }
    select.value = row.role || "skip";
    if (![...select.options].some((option) => option.value === select.value)) select.value = "skip";
    select.addEventListener("change", () => {
      if (select.value === "qr" || select.value === "card") {
        tbody.querySelectorAll("select").forEach((other) => {
          if (other !== select && other.value === select.value) other.value = "skip";
        });
      }
      refreshClassWarn();
    });
    roleCell.append(select);
    tr.append(indexCell, from, nameCell, roleCell);
    tbody.append(tr);
  });
  keepSingleRoles();
  refreshClassWarn();
  document.querySelector("#class-map").showModal();
}

async function saveClassMap() {
  const rows = [...document.querySelectorAll("#class-map-body tr")].map((row) => ({
    index: Number(row.dataset.index),
    source: row.dataset.source,
    display: row.querySelector("input").value,
    role: row.querySelector("select").value,
  }));
  const response = await fetch("/api/model/map", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      path: state.pendingModelPath,
      rows,
      device: document.querySelector("#compute").value || "auto",
      imgsz: Number(document.querySelector("#imgsz").value) || 2048,
      since: state.noteCursor || 0,
    }),
  });
  const body = await response.json();
  if (!response.ok) {
    log(body.error || "Could not save the class names.");
    return;
  }
  document.querySelector("#class-map").close();
  applyModelState(body);
  log(`Detection model ${body.name}`);
  absorbNotes(body);
  if (!rows.some((row) => row.role === "measure")) {
    log("No class is set to extract features. The CSV will have no trait rows.");
  }
  else if (body.error) showBackendError(body.error);
}

function loadDetectionModel() {
  document.querySelector("#model-file").value = state.modelPath || "";
  document.querySelector("#model-load-message").hidden = true;
  document.querySelector("#model-picker").showModal();
}

document.querySelector('#class-map').addEventListener('close', async (event) => {
  if (!state.needsMap || event.target.returnValue !== 'cancel') return;
  const response=await fetch('/api/model/cancel',{method:'POST'});
  if (response.ok) { applyModelState(await response.json()); setBusy(false); }
});
document.querySelector('#class-map').addEventListener('cancel', (event) => { event.target.returnValue='cancel'; });

document.querySelector('#model-file').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') { event.preventDefault(); document.querySelector('#use-model').click(); }
});

document.querySelector("#browse-model").addEventListener("click", async () => {
  const picked = await choosePath("model");
  if (picked?.path) document.querySelector("#model-file").value = picked.path;
});

document.querySelector("#use-model").addEventListener("click", async () => {
  const path = document.querySelector("#model-file").value.trim().replace(/^"|"$/g, "");
  if (!path) return;
  const message = document.querySelector("#model-load-message");
  const controls = [...document.querySelectorAll("#model-picker button, #model-picker input")];
  message.hidden = false;
  message.textContent = /\.pt$/i.test(path) ? "Converting model. This can take a few minutes." : "Loading model…";
  controls.forEach((control) => { control.disabled = true; });
  state.configuring = true;
  setBusy(false);
  try {
    const response = await fetch("/api/model", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path, device: document.querySelector("#compute").value || "auto", imgsz: Number(document.querySelector("#imgsz").value) || 2048, since: state.noteCursor || 0}),
    });
    const body = await response.json();
    if (!response.ok) { message.textContent = body.error || "Could not load that model."; return; }
    document.querySelector("#model-picker").close();
    applyModelState(body);
    if (body.needs_map) openClassMap(body);
    else { log(`Loaded ${body.name}`); absorbNotes(body); }
  } catch (error) { message.textContent = "Could not reach HopBox Infer. Check that the app is still running."; }
  finally {
    controls.forEach((control) => { control.disabled = false; });
    state.configuring = false;
    setBusy(false);
  }
});

async function watchReady() {
  const status = document.querySelector("#status");
  while (true) {
    try {
      const body = await (await fetch("/api/ready")).json();
      state.readyBody = body;
      presentStatus(status, body);
      if (body.ok) {
        try { await refreshVram(true); } catch (error) { /* keep the plan already shown */ }
        return;
      }
    } catch (error) {
      status.textContent = "Waiting for the server";
      status.title = "";
      status.dataset.state = "wait";
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
}
async function loadModelLabel() {
  try {
    const body = await (await fetch("/api/model")).json();
    if (body?.name) applyModelState(body);
    if (body?.needs_map) openClassMap(body);
  } catch (error) {
    /* keep the name already shown */
  }
}

watchReady();
loadModelLabel();
