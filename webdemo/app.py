"""Web demo for the zero-shot audio restoration pipeline.

Run from the repository root:

    python webdemo/app.py

or:

    python app.py

The page accepts an audio upload, restores it with ``src.AudioRestorer`` and
returns the original/restored WAV files plus a few reference-free measurements.
It is intentionally a local demo: files are written to ``webdemo/jobs`` and
removed after one hour.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import AudioQualityMetrics, AudioRestorer, RestorationConfig  # noqa: E402
from src.restoration import SUBTYPES  # noqa: E402
from src.vocals import AMOUNTS as VOCAL_AMOUNTS  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
JOBS_DIR = WEB_DIR / "jobs"
EXAMPLES_DIR = ROOT / "examples"
DOCS_DIR = ROOT / "docs"
GITHUB_URL = (
    "https://github.com/Alessandro1040/"
    "ID-2-Audio-Restoration-for-Generative-Models-Improving-MusicGen-Outputs"
)
SAMPLE_NAME = "demo_dirty.wav"
PRESETS = {
    "music": "Bilanciato",
    "gentle": "Delicato",
    "strong": "Forte",
    "voice": "Voce",
}
VOCAL_MODES = {
    "keep": "Mantieni la voce",
    "reduce": "Riduci la voce",
    "remove": "Rimuovi la voce",
}
MAX_UPLOAD_MB = 200
JOB_TTL_SECONDS = 60 * 60
JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")

app = Flask(
    __name__,
    static_folder=str(WEB_DIR / "static"),
    template_folder=str(WEB_DIR / "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

# The cards shown in the result section.  Values come from AudioQualityMetrics.
CARD_METRICS = (
    ("Noise Floor (dBFS)", "Rumore di fondo", "dBFS", "lower"),
    ("Spectral Flatness", "Piattezza spettrale", "", "lower"),
    ("Clipping (%)", "Clipping", "%", "lower"),
    ("True Peak (dBTP)", "True peak", "dBTP", "neutral"),
    ("Integrated Loudness (LUFS)", "Loudness", "LUFS", "neutral"),
    ("Dynamic Range (dB)", "Dynamic range", "dB", "neutral"),
)
STAGE_LABELS = {
    "declip": "Declip",
    "hum": "Rimozione hum",
    "vocals": "Gestione voce",
    "denoise": "Wiener denoise",
    "highpass": "High-pass",
    "eq": "EQ tonale",
    "silence": "Pulizia silenzi",
    "compressor": "Compressione",
    "dither": "Dither",
}


def _cleanup_jobs() -> None:
    """Delete job folders older than one hour.  Errors are non-fatal."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for folder in JOBS_DIR.glob("*"):
        if not folder.is_dir():
            continue
        try:
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _media_url(job_id: str, filename: str) -> str:
    return f"/media/{job_id}/{filename}"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to ``(samples, sample_rate)``.

    libsndfile covers WAV/FLAC/OGG/AIFF; librosa/audioread is the fallback for
    MP3/M4A and other containers.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception:
        try:
            decoded, sr = librosa.load(str(path), sr=None, mono=False)
        except Exception as exc:
            raise ValueError(
                "Formato audio non riconosciuto o file danneggiato."
            ) from exc
        audio = decoded.T if decoded.ndim > 1 else decoded[:, None]
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("Il file non contiene audio leggibile.")
    if int(sr) <= 0:
        raise ValueError("Sample rate non valido.")
    return audio, int(sr)


def _display_name(raw_name: str | None) -> str:
    name = Path(raw_name or "audio.wav").name
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:160] or "audio.wav"


def _safe_stem(name: str) -> str:
    stem = secure_filename(Path(name).stem) or "audio"
    return stem[:80]


def _write_preview(path: Path, audio: np.ndarray, sr: int) -> None:
    # A browser-safe PCM copy of the input.  The restored file is already WAV.
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def _metric_cards(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for key, label, unit, direction in CARD_METRICS:
        a = _finite(before.get(key))
        b = _finite(after.get(key))
        delta = None if a is None or b is None else b - a
        good = None
        if delta is not None and direction != "neutral":
            good = delta < 0 if direction == "lower" else delta > 0
        cards.append({
            "key": key,
            "label": label,
            "unit": unit,
            "direction": direction,
            "before": a,
            "after": b,
            "delta": delta,
            "good": good,
        })
    return cards


def _stage_list(report: dict[str, Any]) -> list[dict[str, str]]:
    stages = report.get("stages") or {}
    return [
        {"key": key, "label": STAGE_LABELS[key]}
        for key in STAGE_LABELS
        if key in stages
    ]



@app.get("/")
def index():
    sample_available = (EXAMPLES_DIR / SAMPLE_NAME).is_file()
    return render_template(
        "index.html",
        presets=PRESETS,
        vocal_modes=VOCAL_MODES,
        sample_name=SAMPLE_NAME,
        sample_available=sample_available,
        github_url=GITHUB_URL,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@app.get("/examples/<path:filename>")
def example_file(filename: str):
    return send_from_directory(EXAMPLES_DIR, filename, max_age=3600)


@app.get("/docs/<path:filename>")
def docs_file(filename: str):
    return send_from_directory(DOCS_DIR, filename, max_age=3600)


@app.get("/media/<job_id>/<path:filename>")
def media_file(job_id: str, filename: str):
    if not JOB_ID_RE.fullmatch(job_id):
        return jsonify({"error": "Job non valido."}), 404
    folder = _job_dir(job_id)
    if not folder.is_dir():
        return jsonify({"error": "File scaduto o non trovato."}), 404
    return send_from_directory(folder, filename, max_age=600)


@app.post("/api/restore")
def restore():
    _cleanup_jobs()
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"error": "Nessun file ricevuto."}), 400

    preset = request.form.get("preset", "music")
    if preset not in PRESETS:
        return jsonify({"error": "Preset non valido."}), 400

    vocals_mode = request.form.get("vocals", "keep")
    if vocals_mode not in VOCAL_AMOUNTS:
        return jsonify({"error": "Modalità voce non valida."}), 400
    clean_silence = request.form.get("clean_silence", "1") in {"1", "true", "on", "yes"}

    original_name = _display_name(upload.filename)
    suffix = Path(original_name).suffix.lower() or ".audio"
    job_id = uuid.uuid4().hex
    folder = _job_dir(job_id)
    folder.mkdir(parents=True, exist_ok=False)

    try:
        source = folder / f"upload{suffix}"
        upload.save(str(source))
        if source.stat().st_size == 0:
            raise ValueError("Il file caricato è vuoto.")
        audio, sr = _load_audio(source)

        config = RestorationConfig.from_preset(
            preset, bit_depth=24, vocals=vocals_mode,
            clean_silence=clean_silence,
        )
        restorer = AudioRestorer(config=config, verbose=False)
        restored, report = restorer.restore_array(audio, sr, config)
        output_sr = int(report.get("sample_rate", sr))

        before_path = folder / "before.wav"
        after_path = folder / "restored.wav"
        _write_preview(before_path, audio, sr)
        sf.write(
            str(after_path),
            np.clip(restored, -1.0, 1.0),
            output_sr,
            subtype=SUBTYPES[config.bit_depth],
        )

        metrics_before = AudioQualityMetrics(sample_rate=sr).compute_all(audio)
        metrics_after = AudioQualityMetrics(sample_rate=output_sr).compute_all(restored)
        cards = _metric_cards(metrics_before, metrics_after)
        output_name = f"restored_{_safe_stem(original_name)}.wav"

        # The upload is only needed while decoding; don't keep duplicates around.
        try:
            source.unlink()
        except OSError:
            pass

        return jsonify({
            "job_id": job_id,
            "filename": original_name,
            "output_name": output_name,
            "original_url": _media_url(job_id, before_path.name),
            "restored_url": _media_url(job_id, after_path.name),
            "download_url": _media_url(job_id, after_path.name),
            "sample_rate": output_sr,
            "input_sample_rate": sr,
            "channels": int(report.get("channels", 1)),
            "duration_s": _finite(report.get("duration_s")),
            "elapsed_s": _finite(report.get("elapsed_s")),
            "preset": preset,
            "preset_label": PRESETS[preset],
            "vocals": vocals_mode,
            "vocals_label": VOCAL_MODES[vocals_mode],
            "clean_silence": clean_silence,
            "metrics": cards,
            "stages": _stage_list(report),
            "report": report,
        })
    except ValueError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        app.logger.exception("Restoration failed")
        message = str(exc).strip() or exc.__class__.__name__
        return jsonify({"error": f"Impossibile elaborare il file: {message}"}), 400


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({
        "error": f"Il file supera il limite di {MAX_UPLOAD_MB} MB."
    }), 413


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_jobs()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7865"))
    print(f"Audio restoration demo: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
