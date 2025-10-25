"""
Evaluation script for audio restoration

Compares original and restored audio files using objective metrics.

Usage:
    python scripts/evaluate.py --original_dir generated_audio --restored_dir restored_audio
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import AudioQualityMetrics


def evaluate_restoration(
    original_dir: str,
    restored_dir: str,
    output_csv: str = "results/metrics/evaluation_results.csv",
    sample_rate: int = 32000
):
    """
    Evaluate restoration quality by comparing original and restored audio
    
    Args:
        original_dir: Directory with original audio files
        restored_dir: Directory with restored audio files
        output_csv: Path to save results CSV
        sample_rate: Audio sample rate
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Initialize metrics calculator
    metrics_calculator = AudioQualityMetrics(sample_rate=sample_rate)
    
    # Find all original files
    original_path = Path(original_dir)
    original_files = sorted(original_path.glob("*.wav"))
    
    if not original_files:
        print(f"❌ No audio files found in {original_dir}")
        return
    
    print("=" * 70)
    print("AUDIO RESTORATION EVALUATION")
    print("=" * 70)
    print(f"Original directory: {original_dir}")
    print(f"Restored directory: {restored_dir}")
    print(f"Found {len(original_files)} files to evaluate")
    print("=" * 70)
    
    # Store results
    all_results = []
    
    # Evaluate each file pair
    for original_file in tqdm(original_files, desc="Evaluating"):
        # Find corresponding restored file
        restored_filename = f"restored_{original_file.name}"
        restored_file = Path(restored_dir) / restored_filename
        
        if not restored_file.exists():
            print(f"\n⚠️  Restored file not found: {restored_filename}")
            continue
        
        try:
            # Compute metrics
            orig_metrics, rest_metrics, improvements = metrics_calculator.compare_files(
                original_path=str(original_file),
                restored_path=str(restored_file)
            )
            
            # Store results
            for metric_name in orig_metrics.keys():
                all_results.append({
                    'File': original_file.name,
                    'Metric': metric_name,
                    'Original': orig_metrics[metric_name],
                    'Restored': rest_metrics[metric_name],
                    'Delta': improvements[metric_name],
                    'Improvement': improvements[metric_name] > 0 
                                  if metric_name not in ['Spectral Flatness', 'THD (%)', 'Zero Crossing Rate']
                                  else improvements[metric_name] < 0
                })
            
        except Exception as e:
            print(f"\n❌ Error evaluating {original_file.name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        
        avg_original = metric_data['Original'].mean()
        avg_restored = metric_data['Restored'].mean()
        avg_delta = metric_data['Delta'].mean()
        median_delta = metric_data['Delta'].median()
        std_delta = metric_data['Delta'].std()
        
        # Check if improvement
        improved_count = metric_data['Improvement'].sum()
        total_count = len(metric_data)
        improvement_rate = (improved_count / total_count) * 100
        
        print(f"\n{metric}:")
        print(f"  Original (avg):      {avg_original:>10.3f}")
        print(f"  Restored (avg):      {avg_restored:>10.3f}")
        print(f"  Delta (mean):        {avg_delta:>+10.3f}")
        print(f"  Delta (median):      {median_delta:>+10.3f}")
        print(f"  Delta (std):         {std_delta:>10.3f}")
        print(f"  Improvement rate:    {improvement_rate:>10.1f}%")
    
    # Overall improvement summary
    print("\n" + "=" * 70)
    print("OVERALL IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    total_improvements = df['Improvement'].sum()
    total_metrics = len(df)
    overall_rate = (total_improvements / total_metrics) * 100
    
    print(f"Total improvements:  {total_improvements}/{total_metrics}")
    print(f"Overall success rate: {overall_rate:.1f}%")
    
    # Highlight key metrics
    print("\n" + "=" * 70)
    print("KEY METRICS SUMMARY")
    print("=" * 70)
    
    key_metrics = ['SNR (dB)', 'Bandwidth (Hz)', 'Dynamic Range (dB)', 'THD (%)']
    
    print(f"\n{'Metric':<25} {'Avg Improvement':>18}")
    print("-" * 45)
    
    for metric in key_metrics:
        if metric in df['Metric'].values:
            avg_improvement = df[df['Metric'] == metric]['Delta'].mean()
            
            # Format based on metric type
            if 'Hz' in metric:
                print(f"{metric:<25} {avg_improvement:>+15.0f} Hz")
            elif 'dB' in metric:
                print(f"{metric:<25} {avg_improvement:>+15.2f} dB")
            else:
                print(f"{metric:<25} {avg_improvement:>+15.3f}%")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate audio restoration quality"
    )
    
    parser.add_argument(
        '--original_dir',
        type=str,
        required=True,
        help='Directory containing original audio files'
    )
    
    parser.add_argument(
        '--restored_dir',
        type=str,
        required=True,
        help='Directory containing restored audio files'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default='results/metrics/evaluation_results.csv',
        help='Output CSV file for results (default: results/metrics/evaluation_results.csv)'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=32000,
        help='Audio sample rate in Hz (default: 32000)'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.original_dir):
        print(f"❌ Original directory not found: {args.original_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.restored_dir):
        print(f"❌ Restored directory not found: {args.restored_dir}")
        sys.exit(1)
    
    # Run evaluation
    evaluate_restoration(
        original_dir=args.original_dir,
        restored_dir=args.restored_dir,
        output_csv=args.output_csv,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
