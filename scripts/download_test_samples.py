"""
Script per scaricare audio di test da YouTube
"""

import yt_dlp
import os

# Lista di URL YouTube con audio di varie qualità
TEST_URLS = [
    "https://youtu.be/dQw4w9WgXcQ",  # Esempio 1
    "https://youtu.be/9bZkp7q19f0",  # Esempio 2
    # Aggiungi altri 3-8 URL
]

def download_audio(url, output_dir="results/audio_samples"):
    """Scarica audio da YouTube"""
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ar', '44100',
            '-ac', '1',
        ],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        print(f"✅ Downloaded: {url}")

if __name__ == "__main__":
    for url in TEST_URLS:
        download_audio(url)
    print(f"\n✅ Downloaded {len(TEST_URLS)} audio files")
