/**
 * Client-side upload, preview, and result rendering for the Cat vs Dog classifier.
 */

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dzEmpty = document.getElementById("dz-empty");
const dzPreview = document.getElementById("dz-preview");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const btnLabel = predictBtn.querySelector(".primary-btn__label");
const spinner = predictBtn.querySelector(".spinner");
const errorMsg = document.getElementById("error-msg");
const result = document.getElementById("result");
const meters = document.getElementById("meters");
const deviceInfo = document.getElementById("device-info");

const EMOJI = { cat: "\u{1F408}", dog: "\u{1F415}" };
const MAX_BYTES = 8 * 1024 * 1024;

let selectedFile = null;
let previewUrl = null;
let inFlight = false;

/* ---------------------------------------------------------------- helpers */

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

function setLoading(isLoading) {
  inFlight = isLoading;
  predictBtn.disabled = isLoading || !selectedFile;
  btnLabel.textContent = isLoading ? "Analysing" : "Predict";
  spinner.hidden = !isLoading;
  dropzone.style.pointerEvents = isLoading ? "none" : "";
  dropzone.style.opacity = isLoading ? "0.6" : "";
}

function releasePreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

/* --------------------------------------------------------- file selection */

function selectFile(file) {
  clearError();
  result.hidden = true;

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    selectedFile = null;
    showError("That file is not an image. Please choose a JPG, PNG, GIF, WEBP or BMP.");
    syncPreview();
    return;
  }

  if (file.size > MAX_BYTES) {
    selectedFile = null;
    showError("Image is too large. Please pick a file under 8 MB.");
    syncPreview();
    return;
  }

  selectedFile = file;
  releasePreview();
  previewUrl = URL.createObjectURL(file);
  syncPreview();
}

function syncPreview() {
  const hasFile = Boolean(selectedFile);
  dzEmpty.hidden = hasFile;
  dzPreview.hidden = !hasFile;
  predictBtn.disabled = !hasFile || inFlight;

  if (hasFile) {
    previewImg.src = previewUrl;
    previewName.textContent = selectedFile.name;
  } else {
    previewImg.removeAttribute("src");
    previewName.textContent = "";
    fileInput.value = "";
  }
}

function resetAll() {
  releasePreview();
  selectedFile = null;
  syncPreview();
  clearError();
  result.hidden = true;
}

/* ------------------------------------------------------------- prediction */

function renderMeters(probabilities) {
  meters.innerHTML = "";

  Object.entries(probabilities).forEach(([name, value]) => {
    const pct = (value * 100).toFixed(1);

    const meter = document.createElement("div");
    meter.className = `meter meter--${name}`;

    const head = document.createElement("div");
    head.className = "meter__head";

    const label = document.createElement("span");
    label.className = "meter__name";
    label.textContent = name;

    const amount = document.createElement("span");
    amount.className = "meter__value";
    amount.textContent = `${pct}%`;

    head.append(label, amount);

    const track = document.createElement("div");
    track.className = "meter__track";

    const fill = document.createElement("div");
    fill.className = "meter__fill";
    track.append(fill);

    meter.append(head, track);
    meters.append(meter);

    requestAnimationFrame(() => {
      fill.style.width = `${value * 100}%`;
    });
  });
}

function renderResult(data) {
  document.getElementById("verdict-emoji").textContent = EMOJI[data.label] ?? "\u{1F50D}";
  document.getElementById("verdict-label").textContent = data.label;
  document.getElementById("verdict-score").textContent = `${(data.confidence * 100).toFixed(1)}%`;

  renderMeters(data.probabilities);
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runPrediction() {
  if (!selectedFile || inFlight) return;

  clearError();
  setLoading(true);

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json().catch(() => null);

    if (!response.ok || !data) {
      throw new Error(data?.error ?? `Server returned ${response.status}.`);
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "Could not reach the prediction server.");
  } finally {
    setLoading(false);
  }
}

/* ----------------------------------------------------------------- events */

dropzone.addEventListener("click", (event) => {
  if (event.target.closest(".ghost-btn")) return;
  fileInput.click();
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", (event) => {
  selectFile(event.target.files?.[0]);
});

clearBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  resetAll();
});

["dragenter", "dragover"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    if (!inFlight) dropzone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((type) => {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  if (inFlight) return;
  selectFile(event.dataTransfer?.files?.[0]);
});

predictBtn.addEventListener("click", runPrediction);

window.addEventListener("beforeunload", releasePreview);

/* --------------------------------------------------------------- health */

fetch("/api/health")
  .then((response) => (response.ok ? response.json() : Promise.reject()))
  .then((data) => {
    deviceInfo.textContent = `Running on ${data.device} · ${data.classes.length} classes`;
  })
  .catch(() => {
    deviceInfo.textContent = "Model status unavailable";
  });
