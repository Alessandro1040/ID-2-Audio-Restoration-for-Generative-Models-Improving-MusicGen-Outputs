"""Tests for the noise-only gate and the preset defaults around it."""

from __future__ import annotations

import numpy as np

from src import AudioRestorer, RestorationConfig
from src import dsp
from src.restoration import PRESETS
from src.vocals import amount_for


def _tone_and_noise(sr: int = 16000) -> tuple[np.ndarray, int, int, int]:
    """Return (stereo audio, sr, tone_start, tone_end) with a quiet tail."""
    duration = 2.0
    n = int(duration * sr)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.0012, (n, 2))
    t = np.arange(n) / float(sr)
    start, end = int(0.25 * sr), int(0.85 * sr)
    tone = np.zeros(n)
    local = np.arange(end - start) / float(sr)
    # Short attack and a natural-looking decay so the gate sees a real event.
    envelope = np.minimum(1.0, local / 0.01) * np.exp(-local / 0.35)
    tone[start:end] = 0.12 * np.sin(2.0 * np.pi * 440.0 * local) * envelope
    audio = noise.copy()
    audio[:, 0] += tone
    audio[:, 1] -= tone  # out of phase: a mono downmix would erase the note
    return audio, sr, start, end


def _quiet_tonal_tail(sr: int = 16000) -> tuple[np.ndarray, int]:
    """A note whose broadband level falls near the noise floor while ringing."""
    duration = 2.0
    n = int(duration * sr)
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.0012, (n, 2))
    start, end = int(0.25 * sr), int(1.2 * sr)
    local = np.arange(end - start) / float(sr)
    amplitude = 0.010 * np.minimum(1.0, local / 0.005) * np.exp(-local / 0.4)
    tone = amplitude * (np.sin(2.0 * np.pi * 440.0 * local)
                        + 0.3 * np.sin(2.0 * np.pi * 880.0 * local))
    audio = noise.copy()
    audio[start:end, 0] += tone
    audio[start:end, 1] += tone
    return audio, sr


def test_silence_gate_zeros_noise_only_tail() -> None:
    audio, sr, start, end = _tone_and_noise()
    envelope, stats = dsp.silence_envelope(
        audio, sr,
        percentile=10.0, bias=9.5, snr_threshold_db=4.5,
        hold_ms=60.0, release_ms=80.0,
        isolated_hold_ms=20.0, isolated_release_ms=40.0,
    )

    assert envelope[int(0.5 * sr)] > 0.8
    assert envelope[int(0.3 * sr)] > 0.5
    # The event is over well before 1.3 s; from there the envelope must be 0.
    assert np.all(envelope[int(1.3 * sr):] < 1e-9)
    assert stats["muted_ratio"] > 0.35
    assert stats["active_ratio"] > 0.05


def test_silence_gate_detects_out_of_phase_stereo_note() -> None:
    audio, sr, start, end = _tone_and_noise()
    envelope, _ = dsp.silence_envelope(
        audio, sr, percentile=10.0, bias=9.5, snr_threshold_db=4.5,
        hold_ms=60.0, release_ms=80.0,
    )
    # The mono downmix is almost silent, so this fails with the old detector.
    assert envelope[int(0.55 * sr)] > 0.8


def test_tonal_detector_keeps_the_quiet_part_of_a_note_tail() -> None:
    audio, sr = _quiet_tonal_tail()
    common = dict(
        percentile=10.0, bias=9.5, snr_threshold_db=6.0,
        hold_ms=60.0, release_ms=80.0,
        isolated_hold_ms=20.0, isolated_release_ms=40.0,
    )
    with_tonal, stats = dsp.silence_envelope(
        audio, sr, tonal_snr_db=16.0, **common)
    without_tonal, _ = dsp.silence_envelope(
        audio, sr, tonal_snr_db=None, **common)

    # 0.9 s is still inside the ring, but the broadband SNR is below 6 dB.
    assert with_tonal[int(0.9 * sr)] > 0.8
    assert without_tonal[int(0.9 * sr)] < 0.1
    # Once the note has stopped, the gate must close again.
    assert np.all(with_tonal[int(1.5 * sr):] < 1e-9)
    assert stats["tonal_active_ratio"] > 0.1


def test_tonal_detector_does_not_open_on_broadband_noise() -> None:
    sr = 16000
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.01, (2 * sr, 2))
    envelope, stats = dsp.silence_envelope(noise, sr, tonal_snr_db=14.0)

    assert stats["tonal_active_ratio"] < 0.05
    assert stats["muted_ratio"] > 0.95
    assert np.all(envelope < 1e-9)


def test_pipeline_leaves_exact_digital_zeros() -> None:
    audio, sr, _, _ = _tone_and_noise()
    config = RestorationConfig(
        preset="music",
        denoise=False,
        declip=False,
        eq=False,
        normalize=False,
        clean_silence=True,
        bit_depth=32,
        silence_snr_db=4.5,
        silence_hold_ms=60.0,
        silence_release_ms=80.0,
        silence_isolated_hold_ms=20.0,
        silence_isolated_release_ms=40.0,
    )
    restored, report = AudioRestorer(config=config, verbose=False).restore_array(
        audio, sr, config)

    assert np.all(restored[int(1.3 * sr):] == 0.0)
    assert report["stages"]["silence"]["muted_ratio"] > 0.35


def test_all_presets_clean_silence_and_strong_is_stronger() -> None:
    configs = {name: RestorationConfig.from_preset(name) for name in PRESETS}
    for name, config in configs.items():
        assert config.clean_silence is True, name
        assert config.silence_snr_db >= 4.5, name
        assert config.silence_tonal_snr_db == 14.0, name
        assert config.silence_tonal_isolated_release_ms == 250.0, name
    assert configs["strong"].silence_snr_db >= configs["music"].silence_snr_db


def test_vocal_modes_keep_and_remove() -> None:
    assert amount_for("keep") == 0.0
    assert 0.0 < amount_for("reduce") < 1.0
    assert amount_for("remove") > 1.0
