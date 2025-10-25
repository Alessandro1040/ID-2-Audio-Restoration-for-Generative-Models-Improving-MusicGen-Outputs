#!/usr/bin/env python3
"""
Generate audio samples using MusicGen for testing restoration pipeline

Usage:
    python scripts/generate_audio.py --num_samples 10
"""

import argparse
import os
import torch
import torchaudio
from audiocraft.models import MusicGen
from tqdm import tqdm

# Default prompts
PROMPTS = [
    "upbeat electronic dance music with synths and drums",
    "calm piano melody with soft strings background",
    "energetic rock guitar solo with heavy drums",
    "smooth jazz saxophone with double bass",
    "epic orchestral soundtrack with choir",
    "acoustic guitar fingerpicking with light percussion",
    "ambient electronic soundscape with pads",
    "funky bass line with brass section",
    "classical violin concerto with orchestra",
    "lo-fi hip hop beat with vinyl crackle",
]

def main():
    parser = argparse.ArgumentParser(description="Generate audio with MusicGen")
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results/audio_samples/original')
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'])
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading MusicGen model ({args.model_size})...")
    model = MusicGen.get_pretrained(f'facebook/musicgen-{args.model_size}')
    model.set_generation_params(duration=args.duration)
    print(f"Model loaded. Sample rate: {model.sample_rate} Hz\n")
    
    # Generate audio
    prompts = PROMPTS[:args.num_samples]
    
    print(f"Generating {len(prompts)} audio samples...")
    print("=" * 70)
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        output_path = os.path.join(args.output_dir, f"sample_{i+1:03d}.wav")
        
        # Generate
        wav = model.generate([prompt])
        
        # Save
        torchaudio.save(output_path, wav[0].cpu(), model.sample_rate)
        
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        print(f"  → Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print(f"✅ Generation complete! Files saved in {args.output_dir}/")

if __name__ == "__main__":
    main()
