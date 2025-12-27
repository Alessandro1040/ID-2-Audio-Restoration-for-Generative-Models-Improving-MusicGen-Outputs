"""
SCRIPT SEMPLICE PER GENERARE LE IMMAGINI PER IL REPORT LATEX

COSA FA:
- Prende 2 file audio (originale e restaurato)
- Genera 4 immagini PNG per il report
- Le salva in results/plots/

COME USARLO:
    python scripts/generate_report_images.py original.wav restored.wav
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
from pathlib import Path

# Configura stile per immagini belle
plt.style.use('seaborn-v0_8-darkgrid')


def generate_spectrograms(orig_path, rest_path, output_dir):
    """Genera spettrogrammi side-by-side"""
    print("Generating spectrograms...")
    
    # Carica audio
    y_orig, sr = librosa.load(orig_path, sr=44100, mono=True)
    y_rest, _ = librosa.load(rest_path, sr=44100, mono=True)
    
    # Calcola spettrogrammi
    D_orig = librosa.stft(y_orig, n_fft=2048)
    D_rest = librosa.stft(y_rest, n_fft=2048)
    S_orig_db = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
    S_rest_db = librosa.amplitude_to_db(np.abs(D_rest), ref=np.max)
    
    # Crea figura con 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Originale
    img1 = librosa.display.specshow(S_orig_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
    axes[0].set_title('Original Audio - Spectrogram', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Restaurato
    img2 = librosa.display.specshow(S_rest_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title('Restored Audio - Spectrogram', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spectrograms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_dir}/spectrograms_comparison.png")


def generate_waveforms(orig_path, rest_path, output_dir):
    """Genera waveform comparison"""
    print("Generating waveforms...")
    
    # Carica audio
    y_orig, sr = librosa.load(orig_path, sr=44100, mono=True)
    y_rest, _ = librosa.load(rest_path, sr=44100, mono=True)
    
    # Limita a 5 secondi
    max_samples = 5 * sr
    y_orig = y_orig[:max_samples]
    y_rest = y_rest[:max_samples]
    
    time_orig = np.arange(len(y_orig)) / sr
    time_rest = np.arange(len(y_rest)) / sr
    
    # Crea figura
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    axes[0].plot(time_orig, y_orig, linewidth=0.5, color='#e74c3c', alpha=0.8)
    axes[0].set_title('Original Audio', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.0, 1.0)
    
    axes[1].plot(time_rest, y_rest, linewidth=0.5, color='#2ecc71', alpha=0.8)
    axes[1].set_title('Restored Audio', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/waveforms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_dir}/waveforms_comparison.png")


def generate_frequency_response(orig_path, rest_path, output_dir):
    """Genera frequency response comparison"""
    print("Generating frequency response...")
    
    # Carica audio
    y_orig, sr = librosa.load(orig_path, sr=44100, mono=True)
    y_rest, _ = librosa.load(rest_path, sr=44100, mono=True)
    
    # Calcola spettro
    f_orig, Pxx_orig = signal.welch(y_orig, fs=sr, nperseg=4096)
    f_rest, Pxx_rest = signal.welch(y_rest, fs=sr, nperseg=4096)
    
    Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-10)
    Pxx_rest_db = 10 * np.log10(Pxx_rest + 1e-10)
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogx(f_orig, Pxx_orig_db, label='Original', color='#e74c3c', alpha=0.7, linewidth=2)
    ax.semilogx(f_rest, Pxx_rest_db, label='Restored', color='#2ecc71', alpha=0.7, linewidth=2)
    
    ax.set_title('Frequency Response Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_xlim(20, sr/2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frequency_response.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_dir}/frequency_response.png")


def generate_pipeline_diagram(output_dir):
    """Genera schema pipeline"""
    print("Generating pipeline diagram...")
    
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    blocks = [
        (15, "Input Audio", "Low quality, noisy"),
        (13, "Stage 1: Targeted Denoising", "Remove vocal artifacts"),
        (11, "Stage 2: De-clipping", "Repair distortion"),
        (9, "Stage 3: Adaptive Filtering", "Cleanup 50Hz-18kHz"),
        (7, "Stage 4: Spectral Subtraction", "Noise removal"),
        (5, "Stage 5: Clarity Enhancement", "Multi-band EQ"),
        (3, "Stage 6: Final Polish", "Normalize & dither"),
        (1, "Output Audio", "Clean, professional")
    ]
    
    colors = ['#e74c3c', '#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#2ecc71']
    
    for i, (y_pos, title, desc) in enumerate(blocks):
        rect = mpatches.FancyBboxPatch((1, y_pos - 0.6), 8, 1.2, boxstyle="round,pad=0.1",
                                       facecolor=colors[i], edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        ax.text(5, y_pos + 0.2, title, ha='center', va='center', fontsize=13, fontweight='bold', color='white')
        ax.text(5, y_pos - 0.3, desc, ha='center', va='center', fontsize=10, style='italic', color='white')
        
        if i < len(blocks) - 1:
            ax.annotate('', xy=(5, blocks[i+1][0] + 0.6), xytext=(5, y_pos - 0.6),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black', alpha=0.6))
    
    ax.text(5, 16, 'Audio Restoration Pipeline', ha='center', va='top', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_dir}/pipeline_diagram.png")


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_report_images.py <original.wav> <restored.wav>")
        print("\nExample:")
        print("  python scripts/generate_report_images.py original.wav restored.wav")
        sys.exit(1)
    
    orig_path = sys.argv[1]
    rest_path = sys.argv[2]
    
    # Verifica file esistono
    if not Path(orig_path).exists():
        print(f"ERROR: File not found: {orig_path}")
        sys.exit(1)
    if not Path(rest_path).exists():
        print(f"ERROR: File not found: {rest_path}")
        sys.exit(1)
    
    # Crea cartella output
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("  GENERATING IMAGES FOR LATEX REPORT")
    print("="*70 + "\n")
    
    # Genera tutte le immagini
    generate_spectrograms(orig_path, rest_path, output_dir)
    generate_waveforms(orig_path, rest_path, output_dir)
    generate_frequency_response(orig_path, rest_path, output_dir)
    generate_pipeline_diagram(output_dir)
    
    print("\n" + "="*70)
    print(f"  ✓ ALL IMAGES SAVED IN: {output_dir.absolute()}")
    print("="*70)
    print("\nGenerated files:")
    print("  1. spectrograms_comparison.png")
    print("  2. waveforms_comparison.png")
    print("  3. frequency_response.png")
    print("  4. pipeline_diagram.png")
    print("\nUse them in your LaTeX report!")
    print()


if __name__ == "__main__":
    main()
