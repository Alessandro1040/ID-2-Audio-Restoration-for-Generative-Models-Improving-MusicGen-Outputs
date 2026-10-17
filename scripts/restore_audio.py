#!/usr/bin/env python3
"""Restore one or more audio files with the zero-shot pipeline.

Examples
--------
Restore a single MusicGen output (default preset, sample rate preserved):

    python scripts/restore_audio.py output.wav -o restored/

Batch-restore a folder, forcing 32 kHz mono and a -16 LUFS target:

    python scripts/restore_audio.py generated/ -o restored/ \\
        --sample-rate 32000 --mono --target-lufs -16

Disable the tonal EQ and use the conservative noise profile:

    python scripts/restore_audio.py output.wav -o restored/ \\
        --preset gentle --no-eq
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.restoration import PRESETS, AudioRestorer, RestorationConfig, write_report  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {path} does not exist", file=sys.stderr)
    seen = set()
    unique = []
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def build_config(args: argparse.Namespace) -> RestorationConfig:
    overrides = {}
    if args.strength is not None:
        overrides["denoise_strength"] = args.strength
    if args.highpass is not None:
        overrides["highpass_hz"] = args.highpass
    if args.target_lufs is not None:
        overrides["target_lufs"] = args.target_lufs
    config = RestorationConfig.from_preset(
        args.preset,
        target_sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        mono=args.mono,
        true_peak_db=args.true_peak_db,
        **overrides,
    )
    changes = {}
    if args.no_denoise:
        changes["denoise"] = False
    if args.no_declip:
        changes["declip"] = False
    if args.no_eq:
        changes["eq"] = False
    if args.no_normalize:
        changes["normalize"] = False
    if args.hum is not None:
        changes["remove_hum"] = args.hum
    return replace(config, **changes) if changes else config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-shot audio restoration for generative music.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="files or directories")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--preset", default="music", choices=sorted(PRESETS))
    parser.add_argument("--strength", type=float, default=None,
                        help="denoising strength, 0-1.5 (overrides the preset)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="output sample rate; default: preserve the input")
    parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32])
    parser.add_argument("--mono", action="store_true", help="downmix to mono")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="optional loudness target, e.g. -14 or -16")
    parser.add_argument("--true-peak-db", type=float, default=-1.0)
    parser.add_argument("--highpass", type=float, default=None,
                        help="high-pass cutoff in Hz (default: preset, usually off)")
    parser.add_argument("--hum", dest="hum", action="store_true", default=None,
                        help="force mains-hum removal")
    parser.add_argument("--no-hum", dest="hum", action="store_false",
                        help="disable mains-hum removal")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-declip", action="store_true")
    parser.add_argument("--no-eq", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--suffix", default="restored_")
    parser.add_argument("--report", default=None,
                        help="JSON report path (default: <output-dir>/restoration_report.json)")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("error: no audio files found", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    restorer = AudioRestorer(config=config, verbose=not args.quiet)

    if not args.quiet:
        print("=" * 78)
        print("AUDIO RESTORATION")
        print("=" * 78)
        print(f"preset      : {config.preset}")
        print(f"files       : {len(files)}")
        print(f"sample rate : {config.target_sample_rate or 'preserve input'}")
        print(f"output      : {config.bit_depth}-bit {output_dir}")
        print("=" * 78)

    reports = []
    failures = 0
    for index, source in enumerate(files, start=1):
        target = output_dir / f"{args.suffix}{source.stem}.wav"
        if target.exists() and args.no_overwrite:
            if not args.quiet:
                print(f"[{index}/{len(files)}] skip (exists): {target.name}")
            continue
        try:
            report = restorer.restore_file(source, target, config)
            reports.append(report)
            if not args.quiet:
                loudness = report.get("loudness", {})
                before = loudness.get("lufs_before")
                after = loudness.get("lufs_after")
                before_text = "n/a" if before is None else f"{before:.1f}"
                after_text = "n/a" if after is None else f"{after:.1f}"
                print(f"[{index}/{len(files)}] {source.name} -> {target.name} "
                      f"({report['duration_s']:.1f}s, {before_text} -> {after_text} LUFS)")
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(files)}] ERROR {source}: {exc}", file=sys.stderr)

    report_path = Path(args.report) if args.report else output_dir / "restoration_report.json"
    write_report({"config": asdict(config), "files": reports}, report_path)

    if not args.quiet:
        print("=" * 78)
        print(f"done: {len(reports)} restored, {failures} failed")
        print(f"report: {report_path}")
        print("=" * 78)
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())

