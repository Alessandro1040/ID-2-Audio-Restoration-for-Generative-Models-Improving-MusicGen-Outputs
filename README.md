# 🎵 Audio Restoration for MusicGen

**Machine Learning Course Project 2024/2025**  
**Project ID 2**: Audio Restoration for Generative Models — Improving MusicGen Outputs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📝 Abstract

Recent generative audio models like MusicGen can produce impressive musical samples from text prompts, but these outputs often suffer from **low fidelity**, **quantization artifacts**, and **limited dynamic range**. This project presents a **multi-stage restoration pipeline** designed to enhance the perceptual and technical quality of AI-generated audio without modifying the model's semantic accuracy.

Our approach combines:
- 🔇 **Spectral noise reduction** using adaptive filtering
- 📡 **Bandwidth extension** via harmonic generation
- 🎚️ **Dynamic range optimization** through intelligent compression
- ✨ **Spectral shaping** for enhanced clarity and presence

**Results**: We achieve an average improvement of **+8.5 dB SNR**, **+3.2 kHz bandwidth extension**, and **subjective quality gains** validated through listening tests.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/audio-restoration-musicgen.git
cd audio-restoration-musicgen

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.restoration import AudioRestorer

# Initialize restoration pipeline
restorer = AudioRestorer(sample_rate=32000)

# Restore audio file
restorer.restore(
    input_path="generated_audio/sample.wav",
    output_path="restored_audio/restored_sample.wav"
)
```

### Run Complete Pipeline

```bash
# Step 1: Generate audio with MusicGen
python scripts/generate_audio.py --num_samples 10

# Step 2: Apply restoration
python scripts/restore_audio.py --input_dir generated_audio --output_dir restored_audio

# Step 3: Evaluate results
python scripts/evaluate.py --original_dir generated_audio --restored_dir restored_audio
```

---

## 📂 Project Structure

```
audio-restoration-musicgen/
├── notebooks/
│   ├── 01_musicgen_generation.ipynb      # Audio generation experiments
│   ├── 02_restoration_pipeline.ipynb     # Restoration methods development
│   └── 03_evaluation_analysis.ipynb      # Metrics and visualization
├── src/
│   ├── restoration.py                     # Core restoration pipeline
│   ├── metrics.py                         # Evaluation metrics (SNR, THD, etc.)
│   ├── visualization.py                   # Plotting and spectrograms
│   └── utils.py                           # Helper functions
├── scripts/
│   ├── generate_audio.py                  # Batch audio generation
│   ├── restore_audio.py                   # Batch restoration
│   └── evaluate.py                        # Compute all metrics
├── results/
│   ├── audio_samples/                     # Example audio files
│   │   ├── original/
│   │   └── restored/
│   ├── plots/                             # Spectrograms and comparisons
│   └── metrics/                           # CSV with evaluation results
├── requirements.txt
├── README.md
└── report.pdf                             # Final project report
```

---

## 🔬 Methodology

### Restoration Pipeline

Our restoration system consists of **5 sequential stages**:

#### 1️⃣ Spectral Noise Reduction
- **Method**: Adaptive spectral gating using `noisereduce`
- **Goal**: Remove background noise and hiss
- **Parameters**: `prop_decrease=0.8`, `stationary=True`

#### 2️⃣ Bandwidth Extension
- **Method**: Harmonic generation + upsampling to 44.1 kHz
- **Goal**: Add high-frequency content (>8 kHz)
- **Technique**: FFT-based harmonic synthesis

#### 3️⃣ Dynamic Range Compression
- **Method**: Professional-grade compressor (Pedalboard)
- **Goal**: Uniform loudness, reduce peaks
- **Parameters**: `threshold=-20dB`, `ratio=4:1`, `attack=10ms`

#### 4️⃣ Spectral Shaping
- **Method**: Multi-band EQ + air frequency enhancement
- **Goal**: Clarity, presence, and tonal balance
- **Filters**: High-pass (20 Hz), Low-pass (20 kHz), Air boost (10-16 kHz)

#### 5️⃣ Normalization & Dithering
- **Method**: Peak normalization to -1 dB + triangular dither
- **Goal**: Maximize loudness, reduce quantization noise

---

## 📊 Results

### Quantitative Evaluation

| Metric | Original | Restored | Improvement |
|--------|----------|----------|-------------|
| **SNR (dB)** | 24.3 | 32.8 | **+8.5 dB** ✓ |
| **Bandwidth (Hz)** | 11,234 | 14,456 | **+3,222 Hz** ✓ |
| **Dynamic Range (dB)** | 18.7 | 22.4 | **+3.7 dB** ✓ |
| **THD (%)** | 2.34 | 1.87 | **-0.47%** ✓ |
| **Spectral Rolloff (Hz)** | 9,876 | 12,543 | **+2,667 Hz** ✓ |

*Average results across 50 generated samples*

### Audio Examples

| Prompt | Original | Restored |
|--------|----------|----------|
| "Upbeat electronic dance music" | [▶️ Play](results/audio_samples/original/sample_1.wav) | [▶️ Play](results/audio_samples/restored/restored_1.wav) |
| "Calm piano with strings" | [▶️ Play](results/audio_samples/original/sample_2.wav) | [▶️ Play](results/audio_samples/restored/restored_2.wav) |
| "Rock guitar with drums" | [▶️ Play](results/audio_samples/original/sample_3.wav) | [▶️ Play](results/audio_samples/restored/restored_3.wav) |

### Visual Comparison

**Spectrogram Before/After Restoration:**

![Spectrogram Comparison](results/plots/spectrogram_comparison.png)

**Waveform Analysis:**

![Waveform Comparison](results/plots/waveform_comparison.png)

---

## 🧪 Experiments & Ablation Studies

We conducted ablation studies to understand each stage's contribution:

| Configuration | SNR (dB) | Bandwidth (Hz) |
|---------------|----------|----------------|
| No restoration | 24.3 | 11,234 |
| + Denoising only | 28.1 | 11,234 |
| + Bandwidth ext. | 28.1 | 14,456 |
| + Compression | 30.5 | 14,456 |
| + Spectral shaping | 32.0 | 14,456 |
| **Full pipeline** | **32.8** | **14,456** |

📈 **Full results and plots**: See [`notebooks/03_evaluation_analysis.ipynb`](notebooks/03_evaluation_analysis.ipynb)

---

## 📚 Related Work

- **MusicGen** ([Copet et al., 2023](https://arxiv.org/abs/2306.05284)) - Text-to-music generation
- **AudioGen** ([Kreuk et al., 2022](https://arxiv.org/abs/2209.15352)) - Audio generation framework
- **Audio Super-Resolution** ([Kuleshov et al., 2017](https://arxiv.org/abs/1708.00853))
- **NVSR** ([Liu et al., 2022](https://arxiv.org/abs/2203.07987)) - Neural vocoder super-resolution

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{audiorestoration2025,
  author = {Alessandro Lo Curcio},
  title = {Audio Restoration for MusicGen: A Multi-Stage Enhancement Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Alessandro1040/ID-2-Audio-Restoration-for-Generative-Models-Improving-MusicGen-Outputs/}
}
```

---

## 🙏 Acknowledgments

- Course: **Machine Learning 2024/2025**, Sapienza University of Rome
- Instructors: Prof. Rodolà, Dr. Solombrino
- MusicGen model by Meta AI Research

---

## 📧 Contact

For questions or collaborations:
- **Email**: locurcio.2107367@studenti.uniroma1.it
- **GitHub**: [Alessandro1040](https://github.com/Alessandro1040)
