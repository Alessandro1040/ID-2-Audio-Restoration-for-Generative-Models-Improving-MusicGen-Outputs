(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("upload-form");
  const input = $("file-input");
  const dropzone = $("dropzone");
  const fileRow = $("file-row");
  const fileName = $("file-name");
  const fileSize = $("file-size");
  const removeButton = $("remove-file");
  const restoreButton = $("restore-button");
  const status = $("form-status");
  const result = $("result");
  const beforeAudio = $("before-audio");
  const afterAudio = $("after-audio");
  const beforeName = $("before-name");
  const afterName = $("after-name");
  const downloadLink = $("download-link");
  const metricsGrid = $("metrics-grid");
  const metricsTime = $("metrics-time");
  const stageChips = $("stage-chips");
  const reportJson = $("report-json");

  const sampleAudio = document.querySelector("[data-sample-audio]");
  const sampleUrl = sampleAudio ? sampleAudio.getAttribute("src") : "/examples/demo_dirty.wav";

  let selectedFile = null;
  let busy = false;
  let timer = null;

  function formatNumber(value, digits) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(digits).replace(".", ",");
  }

  function formatMetric(value, unit, signed) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    let digits = 2;
    if (Math.abs(number) >= 100) digits = 0;
    else if (Math.abs(number) >= 10) digits = 1;
    let text = formatNumber(number, digits);
    if (signed && number > 0) text = "+" + text;
    return unit ? text + " " + unit : text;
  }

  function formatDelta(value, unit) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return formatMetric(value, unit, true);
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }
    const digits = index === 0 ? 0 : value >= 10 ? 1 : 2;
    return value.toFixed(digits).replace(".", ",") + " " + units[index];
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(Number(seconds)) || Number(seconds) <= 0) return "—";
    const total = Math.round(Number(seconds));
    const minutes = Math.floor(total / 60);
    const rest = total % 60;
    return minutes + ":" + String(rest).padStart(2, "0");
  }

  function channelLabel(channels) {
    const count = Number(channels);
    if (count === 1) return "mono";
    if (count === 2) return "stereo";
    return count ? count + " canali" : "—";
  }

  function setStatus(message, tone) {
    status.textContent = message || "";
    status.classList.toggle("is-error", tone === "error");
    status.classList.toggle("is-ok", tone === "ok");
  }

  function chooseFile(file) {
    if (!file) return;
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileRow.hidden = false;
    dropzone.classList.add("has-file");
    $("drop-title").textContent = "File pronto per la pulizia";
    setStatus("File selezionato. Scegli la modalità e premi “Ripulisci audio”.", "");
  }

  function clearFile() {
    selectedFile = null;
    input.value = "";
    fileRow.hidden = true;
    dropzone.classList.remove("has-file");
    $("drop-title").textContent = "Trascina un file audio qui";
    setStatus("", "");
  }

  input.addEventListener("change", function () {
    const file = input.files && input.files[0];
    if (file) chooseFile(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, function (event) {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) chooseFile(file);
  });

  removeButton.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();
    clearFile();
  });

  document.querySelectorAll("[data-load-sample]").forEach((button) => {
    button.addEventListener("click", async function () {
      setStatus("Carico l'esempio…", "");
      try {
        const response = await fetch(sampleUrl, { cache: "no-store" });
        if (!response.ok) throw new Error("Esempio non disponibile.");
        const blob = await response.blob();
        const file = new File([blob], "demo_dirty.wav", { type: blob.type || "audio/wav" });
        chooseFile(file);
        document.getElementById("studio").scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        setStatus("Non riesco a caricare l'esempio: " + error.message, "error");
      }
    });
  });

  function renderMetricCard(metric) {
    const card = document.createElement("article");
    card.className = "metric-card";

    const label = document.createElement("div");
    label.className = "metric-card__label";
    const labelText = document.createElement("span");
    labelText.textContent = metric.label;
    label.appendChild(labelText);

    const values = document.createElement("div");
    values.className = "metric-card__values";
    const signed = metric.key === "True Peak (dBTP)";
    const before = document.createElement("span");
    before.className = "metric-card__before";
    before.textContent = formatMetric(metric.before, metric.unit, signed);
    const after = document.createElement("span");
    after.className = "metric-card__after";
    after.textContent = formatMetric(metric.after, metric.unit, signed);
    values.appendChild(before);
    values.appendChild(after);

    const delta = document.createElement("span");
    const isNeutral = metric.direction === "neutral";
    delta.className = "metric-card__delta " + (isNeutral ? "metric-card__delta--neutral" :
      metric.good ? "metric-card__delta--good" : "metric-card__delta--bad");
    delta.textContent = formatDelta(metric.delta, metric.unit);

    const hint = document.createElement("p");
    hint.className = "metric-card__hint";
    hint.textContent = isNeutral ? "descrittore neutro" :
      metric.direction === "lower" ? "più basso è meglio" : "più alto è meglio";

    card.appendChild(label);
    card.appendChild(values);
    card.appendChild(delta);
    card.appendChild(hint);
    return card;
  }

  function renderResult(data) {
    result.hidden = false;
    requestAnimationFrame(() => result.classList.add("is-visible"));

    beforeAudio.src = data.original_url;
    afterAudio.src = data.restored_url;
    beforeAudio.load();
    afterAudio.load();
    beforeName.textContent = data.filename;
    afterName.textContent = data.output_name;
    downloadLink.href = data.download_url;
    downloadLink.setAttribute("download", data.output_name);

    $("result-filename").textContent = data.filename;
    const extras = [];
    if (data.vocals) extras.push("voce: " + data.vocals_label);
    if (data.clean_silence) extras.push("silenzi puliti");
    $("result-meta").textContent = [data.preset_label].concat(extras, [
      channelLabel(data.channels),
      formatNumber(data.sample_rate, 0) + " Hz"
    ]).join(" · ");
    $("meta-sr").textContent = formatNumber(data.sample_rate, 0) + " Hz";
    $("meta-channels").textContent = channelLabel(data.channels);
    $("meta-duration").textContent = formatDuration(data.duration_s);
    $("meta-elapsed").textContent = data.elapsed_s == null ? "—" : formatNumber(data.elapsed_s, 2) + " s";

    metricsGrid.replaceChildren.apply(metricsGrid, (data.metrics || []).map(renderMetricCard));
    metricsTime.textContent = data.elapsed_s == null ? "" : "pipeline: " + formatNumber(data.elapsed_s, 2) + " s";

    stageChips.replaceChildren();
    (data.stages || []).forEach((stage) => {
      const chip = document.createElement("span");
      chip.className = "stage-chip";
      chip.textContent = stage.label;
      stageChips.appendChild(chip);
    });
    reportJson.textContent = JSON.stringify(data.report || {}, null, 2);
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    if (busy) return;
    if (!selectedFile) {
      setStatus("Scegli prima un file audio (o prova l'esempio).", "error");
      input.focus();
      return;
    }

    const preset = form.querySelector('input[name="preset"]:checked');
    const vocalMode = form.querySelector('input[name="vocals"]:checked');
    const cleanSilence = form.querySelector('input[name="clean_silence"]');
    const data = new FormData();
    data.append("audio", selectedFile, selectedFile.name);
    data.append("preset", preset ? preset.value : "music");
    data.append("vocals", vocalMode ? vocalMode.value : "keep");
    data.append("clean_silence", cleanSilence && cleanSilence.checked ? "1" : "0");

    busy = true;
    restoreButton.disabled = true;
    restoreButton.classList.add("is-busy");
    restoreButton.querySelector(".button-label").textContent = "Sto ripulendo…";
    setStatus("Elaborazione in corso. File lunghi possono richiedere qualche minuto…", "");

    const started = performance.now();
    timer = window.setInterval(function () {
      const elapsed = (performance.now() - started) / 1000;
      setStatus("Elaborazione in corso · " + formatNumber(elapsed, 0) + " s", "");
    }, 1000);

    try {
      const response = await fetch("/api/restore", { method: "POST", body: data });
      let payload = null;
      try {
        payload = await response.json();
      } catch (_error) {
        payload = null;
      }
      if (!response.ok) {
        throw new Error((payload && payload.error) || "Errore del server (" + response.status + ").");
      }
      renderResult(payload);
      setStatus("Fatto. Il risultato è pronto qui sotto.", "ok");
    } catch (error) {
      setStatus(error.message || "Qualcosa è andato storto.", "error");
    } finally {
      window.clearInterval(timer);
      timer = null;
      busy = false;
      restoreButton.disabled = false;
      restoreButton.classList.remove("is-busy");
      restoreButton.querySelector(".button-label").textContent = "Ripulisci audio";
    }
  });

  // Only one player at a time: avoids two tracks overlapping during an A/B listen.
  [beforeAudio, afterAudio].forEach((player) => {
    player.addEventListener("play", function () {
      [beforeAudio, afterAudio].forEach((other) => {
        if (other !== player && !other.paused) other.pause();
      });
    });
  });
})();
