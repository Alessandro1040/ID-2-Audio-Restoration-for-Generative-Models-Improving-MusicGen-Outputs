# ===================================================================
# CELLA PER GENERARE IMMAGINI PER IL REPORT LATEX
# ===================================================================

# 1. Carica i tuoi file audio
from google.colab import files
import shutil

print("📁 Carica il file ORIGINAL (mp3 o wav):")
uploaded = files.upload()
original_filename = list(uploaded.keys())[0]
shutil.move(original_filename, 'original_upload.wav')

print("\n📁 Carica il file RESTORED (mp3 o wav):")
uploaded = files.upload()
restored_filename = list(uploaded.keys())[0]
shutil.move(restored_filename, 'restored_upload.wav')

print("\n✓ File caricati!")

# 2. Converte in WAV se necessario (usando librosa)
import librosa
import soundfile as sf

y_orig, sr = librosa.load('original_upload.wav', sr=44100, mono=True)
y_rest, _ = librosa.load('restored_upload.wav', sr=44100, mono=True)

sf.write('original.wav', y_orig, 44100)
sf.write('restored.wav', y_rest, 44100)

print("✓ File convertiti in WAV a 44100 Hz")

# 3. GENERA LE IMMAGINI
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy import signal
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-darkgrid')

# Crea cartella per i plot
!mkdir -p results/plots

print("\n🎨 Generazione immagini in corso...\n")

# --------------- IMMAGINE 1: SPETTROGRAMMI ---------------
print("1/4 Generating spectrograms...")
y_orig, sr = librosa.load('original.wav', sr=44100, mono=True)
y_rest, _ = librosa.load('restored.wav', sr=44100, mono=True)

D_orig = librosa.stft(y_orig, n_fft=2048)
D_rest = librosa.stft(y_rest, n_fft=2048)
S_orig_db = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
S_rest_db = librosa.amplitude_to_db(np.abs(D_rest), ref=np.max)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

img1 = librosa.display.specshow(S_orig_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
axes[0].set_title('Original Audio - Spectrogram', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

img2 = librosa.display.specshow(S_rest_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
axes[1].set_title('Restored Audio - Spectrogram', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('results/plots/spectrograms_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ spectrograms_comparison.png")

# --------------- IMMAGINE 2: WAVEFORMS ---------------
print("2/4 Generating waveforms...")
max_samples = 5 * sr
y_orig_short = y_orig[:max_samples]
y_rest_short = y_rest[:max_samples]

time_orig = np.arange(len(y_orig_short)) / sr
time_rest = np.arange(len(y_rest_short)) / sr

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(time_orig, y_orig_short, linewidth=0.5, color='#e74c3c', alpha=0.8)
axes[0].set_title('Original Audio', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.0, 1.0)

axes[1].plot(time_rest, y_rest_short, linewidth=0.5, color='#2ecc71', alpha=0.8)
axes[1].set_title('Restored Audio', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Amplitude', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1.0, 1.0)

plt.tight_layout()
plt.savefig('results/plots/waveforms_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ waveforms_comparison.png")

# --------------- IMMAGINE 3: FREQUENCY RESPONSE ---------------
print("3/4 Generating frequency response...")
f_orig, Pxx_orig = signal.welch(y_orig, fs=sr, nperseg=4096)
f_rest, Pxx_rest = signal.welch(y_rest, fs=sr, nperseg=4096)

Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-10)
Pxx_rest_db = 10 * np.log10(Pxx_rest + 1e-10)

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
plt.savefig('results/plots/frequency_response.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ frequency_response.png")

# --------------- IMMAGINE 4: PIPELINE DIAGRAM ---------------
print("4/4 Generating pipeline diagram...")
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
plt.savefig('results/plots/pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ pipeline_diagram.png")

print("\n" + "="*70)
print("✅ TUTTE LE IMMAGINI GENERATE!")
print("="*70)
print("\nFile generati in results/plots/:")
print("  1. spectrograms_comparison.png")
print("  2. waveforms_comparison.png")
print("  3. frequency_response.png")
print("  4. pipeline_diagram.png")

# 4. SCARICA LE IMMAGINI
print("\n📥 Download delle immagini...")
!zip -r report_images.zip results/plots/
files.download('report_images.zip')

print("\n✓ Fatto! Scarica il file report_images.zip e usalo nel tuo LaTeX")
