#!/usr/bin/env python3
"""Evaluate restoration quality against the original (and optionally clean) audio.

Use ``--reference_dir`` to enable SI-SDR, SNR, LSD, PESQ and STOI; without a
reference only the reference-free descriptors are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import AudioQualityMetrics, METRIC_DIRECTION  # noqa: E402


def find_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(p for p in directory.rglob(pattern) if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate audio restoration")
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--restored_dir", required=True)
    parser.add_argument("--reference_dir", default=None,
                        help="optional folder with the clean reference files")
    parser.add_argument("--output_csv", default="results/metrics/evaluation_results.csv")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--prefix", default="restored_",
                        help="prefix added to the restored filenames")
    parser.add_argument("--json", default=None, help="optional JSON summary path")
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    restored_dir = Path(args.restored_dir)
    if not original_dir.is_dir():
        print(f"error: {original_dir} is not a directory", file=sys.stderr)
        return 1

    originals = find_files(original_dir, args.pattern)
    if not originals:
        print(f"error: no files matching {args.pattern} in {original_dir}", file=sys.stderr)
        return 1

    metrics_calc = AudioQualityMetrics(sample_rate=args.sample_rate)
    rows = []
    summary: dict[str, dict] = {}
    missing = 0

    print("=" * 78)
    print("AUDIO RESTORATION EVALUATION")
    print("=" * 78)
    print(f"files: {len(originals)} | reference: {args.reference_dir or 'none'}")
    print("=" * 78)

    for original in originals:
        restored = restored_dir / f"{args.prefix}{original.name}"
        if not restored.exists():
            restored = restored_dir / original.name
        if not restored.exists():
            print(f"warning: restored file not found for {original.name}")
            missing += 1
            continue
        reference = None
        if args.reference_dir:
            candidate = Path(args.reference_dir) / original.name
            reference = str(candidate) if candidate.exists() else None

        original_metrics, restored_metrics, improvements = metrics_calc.compare_files(
            str(original), str(restored), reference)
        for name, value in original_metrics.items():
            restored_value = restored_metrics.get(name)
            delta = improvements.get(name)
            verdict = "n/a"
            good = metrics_calc.is_improvement(name, delta)
            if good is not None:
                verdict = "improved" if good else "worse"
            elif delta is not None:
                verdict = "neutral"
            rows.append({
                "File": original.name,
                "Metric": name,
                "Original": value,
                "Restored": restored_value,
                "Delta": delta,
                "Direction": METRIC_DIRECTION.get(name, "neutral"),
                "Verdict": verdict,
            })
            bucket = summary.setdefault(name, {"deltas": [], "improved": 0,
                                               "worse": 0, "neutral": 0})
            if delta is not None:
                bucket["deltas"].append(delta)
                if good is True:
                    bucket["improved"] += 1
                elif good is False:
                    bucket["worse"] += 1
                else:
                    bucket["neutral"] += 1

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "File", "Metric", "Original", "Restored", "Delta", "Direction", "Verdict"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_csv}")
    print("\n" + "=" * 78)
    print("SUMMARY (mean delta; counts over files)")
    print("=" * 78)
    for name, bucket in summary.items():
        if not bucket["deltas"]:
            continue
        mean = sum(bucket["deltas"]) / len(bucket["deltas"])
        print(f"{name:<30} mean {mean:+9.3f} | "
              f"improved {bucket['improved']:3d}  worse {bucket['worse']:3d}  "
              f"neutral {bucket['neutral']:3d}")
    print("=" * 78)

    if args.json:
        payload = {
            "files": len(originals),
            "missing": missing,
            "sample_rate": args.sample_rate,
            "reference_dir": args.reference_dir,
            "summary": {
                name: {
                    "mean_delta": sum(b["deltas"]) / len(b["deltas"]),
                    "improved": b["improved"],
                    "worse": b["worse"],
                    "neutral": b["neutral"],
                }
                for name, b in summary.items() if b["deltas"]
            },
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"JSON summary: {path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
