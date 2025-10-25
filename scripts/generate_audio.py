"""
Batch audio generation script using MusicGen

Generates audio samples from text prompts for testing restoration pipeline.

Usage:
    python scripts/generate_audio.py --num_samples 10 --output_dir generated_audio
"""

import argparse
import os
import torch
import torchaudio
from audiocraft.models import MusicGen
from tqdm import tqdm


# Default prompts for diverse audio generation
DEFAULT_PROMPTS = [
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
    "heavy metal guitar riffs with fast drums",
    "reggae rhythm with steel drums",
    "country music with banjo and harmonica",
    "techno beat with synthesizer arpeggios",
    "bossa nova guitar with light percussion",
    "dubstep drop with heavy bass wobbles",
    "flamenco guitar with handclaps",
    "gospel choir with piano accompaniment",
    "trap beat with 808 bass and hi-hats",
    "bluegrass banjo with fiddle"
]


def generate_audio_samples(
    num_samples: int = 10,
    output_dir: str = "generated_audio",
    model_size: str = "small",
    duration: int = 10,
    temperature: float = 1.0,
    top_k: int = 250,
    custom_prompts: list = None
):
    """
    Generate audio samples using MusicGen
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save generated audio
        model_size: MusicGen model size ('small', 'medium', 'large')
        duration: Duration of each sample in seconds
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        custom_prompts: Optional list of custom prompts
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MusicGen model
    print(f"Loading MusicGen model ({model_size})...")
    model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k
    )
    
    print(f"Model loaded. Sample rate: {model.sample_rate} Hz")
    
    # Select prompts
    if custom_prompts:
        prompts = custom_prompts[:num_samples]
    else:
        prompts = DEFAULT_PROMPTS[:num_samples]
    
    # Generate audio samples
    print(f"\nGenerating {num_samples} audio samples...")
    print("=" * 70)
    
    generated_files = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        print(f"\n[{i+1}/{num_samples}] Prompt: {prompt}")
        
        try:
            # Generate audio
            wav = model.generate([prompt])  # Shape: [1, 1, samples]
            
            # Save to file
            output_path = os.path.join(output_dir, f"sample_{i+1:03d}.wav")
            torchaudio.save(
                output_path,
                wav[0].cpu(),
                sample_rate=model.sample_rate
            )
            
            generated_files.append(output_path)
            print(f"  ✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ Generation complete!")
    print(f"Generated {len(generated_files)}/{num_samples} samples")
    print(f"Output directory: {output_dir}/")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio samples using MusicGen"
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=10,
        help='Number of samples to generate (default: 10)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='generated_audio',
        help='Output directory for generated audio (default: generated_audio)'
    )
    
    parser.add_argument(
        '--model_size', 
        type=str, 
        default='small',
        choices=['small', 'medium', 'large'],
        help='MusicGen model size (default: small)'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=10,
        help='Duration of each sample in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=250,
        help='Top-k sampling parameter (default: 250)'
    )
    
    parser.add_argument(
        '--prompts_file',
        type=str,
        default=None,
        help='Path to text file with custom prompts (one per line)'
    )
    
    args = parser.parse_args()
    
    # Load custom prompts if provided
    custom_prompts = None
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            custom_prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(custom_prompts)} custom prompts from {args.prompts_file}")
    
    # Generate audio
    generate_audio_samples(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_size=args.model_size,
        duration=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        custom_prompts=custom_prompts
    )


if __name__ == "__main__":
    main()
