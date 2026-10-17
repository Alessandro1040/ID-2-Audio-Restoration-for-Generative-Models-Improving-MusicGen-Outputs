# Demo audio

`demo_dirty.wav` is the degraded clip produced by `python scripts/make_demo.py`
(12 s, 32 kHz, mono, 24-bit). It contains broadband hiss at roughly 12 dB SNR,
8-bit quantization, an 8 kHz band limit and a few clipped peaks.

It is committed because it is small (about 1.1 MB), the repository's
`.gitignore` explicitly whitelists `examples/*.wav`, and the web demo uses it
for the "Prova l'esempio" button. The clean and restored versions stay in
`results/demo/` and are generated locally.
