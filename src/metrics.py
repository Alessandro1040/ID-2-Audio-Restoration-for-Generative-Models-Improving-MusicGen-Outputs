"""
Audio Quality Metrics for Evaluation

Implements various objective metrics to assess audio quality improvements.

Author: Alessandro Lo Curcio
Date: 2025
"""

import numpy as np
import librosa
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioQualityMetrics:
    """
    Compute objective audio quality metrics
    
    Metrics include:
        - Signal-to-Noise Ratio (SNR)
        - Spectral Flatness
        - Bandwidth
        - Spectral Rolloff
        - Dynamic Range
        - Total Harmonic Distortion (THD)
        - Spectral Centroid
        - Zero Crossing Rate
    
    Args:
        sample_rate (int): Audio sample rate in Hz
    
    Example:
        >>> metrics = AudioQualityMetrics(sample_rate=32000)
        >>> results = metrics._all(audio_signal)
    """
    
    def __init__(self, sample_rate: int = 32000):
        self.sr = sample_rate
    
    def compute_snr(self, original: np.ndarray, restored: np.ndarray) -> float:
        """
        Compute the Signal-to-Noise Ratio (SNR) between original and restored audio.
        Uses energy alignment to avoid volume bias in the measurement.
        
        Args:
            original: Reference/Original audio signal.
            restored: Processed/Restored audio signal.
            
        Returns:
            SNR value in decibels (dB).
        """
        # Ensure signals have the same length
        target_len = min(len(original), len(restored))
        original = original[:target_len]
        restored = restored[:target_len]
        
        # Energy normalization for noise computation
        original_rms = np.sqrt(np.mean(original**2))
        restored_rms = np.sqrt(np.mean(restored**2))
        
        if restored_rms == 0: 
            return 0.0
        
        # Scale restored signal to match original energy levels before computing noise
        restored_scaled = restored * (original_rms / restored_rms)
        noise = original - restored_scaled
        
        signal_power = np.mean(original**2)
        noise_power = np.mean(noise**2)
        
        # Handle identical signals or near-zero noise to avoid division by zero
        if noise_power < 1e-10: 
            return 100.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """
        Compute Spectral Flatness
        
        Lower values indicate more tonal content (better for music)
        Values close to 1 indicate noise-like spectrum
        
        Args:
            audio: Input audio signal
        
        Returns:
            Spectral flatness (0-1)
        """
        flatness = librosa.feature.spectral_flatness(y=audio)
        return float(np.mean(flatness))
    
    def compute_bandwidth(self, audio: np.ndarray) -> float:
        """
        Compute effective bandwidth in Hz
        
        Measures the spread of the spectrum
        
        Args:
            audio: Input audio signal
        
        Returns:
            Bandwidth in Hz
        """
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        return float(np.mean(bandwidth))
    
    def compute_spectral_rolloff(self, audio: np.ndarray, 
                                 roll_percent: float = 0.85) -> float:
        """
        Compute Spectral Roll-off frequency
        
        Frequency below which `roll_percent` of energy is contained
        Higher values indicate more high-frequency content
        
        Args:
            audio: Input audio signal
            roll_percent: Percentage of energy (default 0.85 = 85%)
        
        Returns:
            Roll-off frequency in Hz
        """
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=self.sr, 
            roll_percent=roll_percent
        )
        return float(np.mean(rolloff))
    
    def compute_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Compute Dynamic Range in dB
        
        Difference between peak and RMS level
        
        Args:
            audio: Input audio signal
        
        Returns:
            Dynamic range in dB
        """
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        dr = 20 * np.log10(peak / (rms + 1e-10))
        return float(dr)
    
    def compute_thd(self, audio: np.ndarray, n_harmonics: int = 5) -> float:
        """
        Compute Total Harmonic Distortion (THD)
        
        Ratio of harmonic power to fundamental power
        Lower values indicate less distortion
        
        Args:
            audio: Input audio signal
            n_harmonics: Number of harmonics to consider
        
        Returns:
            THD as percentage
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        
        # Find fundamental frequency (skip DC component)
        fundamental_idx = np.argmax(magnitude[10:]) + 10
        fundamental_power = magnitude[fundamental_idx] ** 2
        
        # Sum harmonic powers
        harmonic_power = 0
        for n in range(2, n_harmonics + 1):
            harmonic_idx = int(fundamental_idx * n)
            if harmonic_idx < len(magnitude):
                harmonic_power += magnitude[harmonic_idx] ** 2
        
        # Compute THD
        thd = np.sqrt(harmonic_power / (fundamental_power + 1e-10))
        
        return float(thd * 100)  # Convert to percentage
    
    def compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """
        Compute Spectral Centroid in Hz
        
        Center of mass of the spectrum
        Indicates "brightness" of sound
        
        Args:
            audio: Input audio signal
        
        Returns:
            Spectral centroid in Hz
        """
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        return float(np.mean(centroid))
    
    def compute_zcr(self, audio: np.ndarray) -> float:
        """
        Compute Zero Crossing Rate
        
        Number of times signal crosses zero
        Related to noisiness of signal
        
        Args:
            audio: Input audio signal
        
        Returns:
            Zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(audio)
        return float(np.mean(zcr))
    
    def compute_all(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            audio: Input audio signal
        
        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'SNR (dB)': self.compute_snr(audio),
            'Spectral Flatness': self.compute_spectral_flatness(audio),
            'Bandwidth (Hz)': self.compute_bandwidth(audio),
            'Spectral Rolloff (Hz)': self.compute_spectral_rolloff(audio),
            'Dynamic Range (dB)': self.compute_dynamic_range(audio),
            'THD (%)': self.compute_thd(audio),
            'Spectral Centroid (Hz)': self.compute_spectral_centroid(audio),
            'Zero Crossing Rate': self.compute_zcr(audio)
        }
        
        return metrics
    
    def evaluate_file(self, filepath: str) -> Dict[str, float]:
        """
        Evaluate audio file and return all metrics
        
        Args:
            filepath: Path to audio file
        
        Returns:
            Dictionary with metric values
        """
        audio, sr = librosa.load(filepath, sr=self.sr, mono=True)
        return self.compute_all(audio)
    
    def compare_files(self, 
                     original_path: str,
                     restored_path: str) -> Tuple[Dict, Dict, Dict]:
        """
        Compare original and restored audio files
        
        Args:
            original_path: Path to original audio
            restored_path: Path to restored audio
        
        Returns:
            Tuple of (original_metrics, restored_metrics, improvements)
        """
        # Compute metrics for both files
        original_metrics = self.evaluate_file(original_path)
        restored_metrics = self.evaluate_file(restored_path)
        
        # Compute improvements (delta)
        improvements = {}
        for key in original_metrics.keys():
            improvements[key] = restored_metrics[key] - original_metrics[key]
        
        return original_metrics, restored_metrics, improvements
    
    def print_comparison(self,
                        original_metrics: Dict,
                        restored_metrics: Dict,
                        improvements: Dict):
        """
        Print formatted comparison table
        
        Args:
            original_metrics: Metrics for original audio
            restored_metrics: Metrics for restored audio
            improvements: Delta between restored and original
        """
        print("\n" + "="*70)
        print(f"{'Metric':<30} {'Original':>12} {'Restored':>12} {'Delta':>12}")
        print("="*70)
        
        for metric_name in original_metrics.keys():
            orig = original_metrics[metric_name]
            rest = restored_metrics[metric_name]
            delta = improvements[metric_name]
            
            # Determine if improvement is good
            # For most metrics, higher is better
            # Exceptions: Spectral Flatness, THD, ZCR (lower is better)
            lower_is_better = metric_name in [
                'Spectral Flatness', 
                'THD (%)', 
                'Zero Crossing Rate'
            ]
            
            is_improvement = delta < 0 if lower_is_better else delta > 0
            indicator = "✓" if is_improvement else "✗"
            
            # Format numbers
            if 'Hz' in metric_name:
                print(f"{metric_name:<30} {orig:>12.0f} {rest:>12.0f} "
                      f"{delta:>+12.0f} {indicator}")
            else:
                print(f"{metric_name:<30} {orig:>12.3f} {rest:>12.3f} "
                      f"{delta:>+12.3f} {indicator}")
        
        print("="*70)


def main():
    """Example usage"""
    metrics = AudioQualityMetrics(sample_rate=32000)
    
    # Single file evaluation
    results = metrics.evaluate_file("generated_audio/sample_1.wav")
    print("\nMetrics for sample_1.wav:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    # Compare original vs restored
    print("\n" + "="*70)
    print("COMPARISON: Original vs Restored")
    print("="*70)
    
    orig, rest, improvements = metrics.compare_files(
        original_path="generated_audio/sample_1.wav",
        restored_path="restored_audio/restored_sample_1.wav"
    )
    
    metrics.print_comparison(orig, rest, improvements)


if __name__ == "__main__":
    main()
