#!/usr/bin/env python3
"""
Evaluate restoration quality with metrics

Usage:
    python scripts/evaluate.py --original_dir results/audio_samples/original --restored_dir results/audio_samples/restored
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import AudioQualityMetrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio restoration")
    parser.add_argument('--original_dir', type=str, required=True)
    parser.add_argument('--restored_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='results/metrics/evaluation_results.csv')
    parser.add_argument('--sample_rate', type=int, default=32000)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Initialize metrics
    metrics_calc = AudioQualityMetrics(sample_rate=args.sample_rate)
    
    # Find files
    original_files = sorted(Path(args.original_dir).glob("*.wav"))
    
    print("=" * 70)
    print("AUDIO RESTORATION EVALUATION")
    print("=" * 70)
    print(f"Files to evaluate: {len(original_files)}")
    print("=" * 70 + "\n")
    
    # Evaluate each pair
    all_results = []
    
    for orig_file in tqdm(original_files, desc="Evaluating"):
        rest_file = Path(args.restored_dir) / f"restored_{orig_file.name}"
        
        if not rest_file.exists():
            print(f"⚠️  Skipping {orig_file.name}: restored file not found")
            continue
        
        try:
            orig_metrics, rest_metrics, improvements = metrics_calc.compare_files(
                str(orig_file), str(rest_file)
            )
            
            for metric_name in orig_metrics.keys():
                all_results.append({
                    'File': orig_file.name,
                    'Metric': metric_name,
                    'Original': orig_metrics[metric_name],
                    'Restored': rest_metrics[metric_name],
                    'Delta': improvements[metric_name]
                })
                
        except Exception as e:
            print(f"❌ Error: {orig_file.name}: {e}")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Results saved to: {args.output_csv}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for metric in df['Metric'].unique():
        data = df[df['Metric'] == metric]
        avg_delta = data['Delta'].mean()
        print(f"\n{metric}:")
        print(f"  Average improvement: {avg_delta:+.3f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
