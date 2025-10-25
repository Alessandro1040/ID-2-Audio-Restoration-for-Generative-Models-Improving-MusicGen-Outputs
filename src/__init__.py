"""
Audio Restoration for MusicGen
Multi-stage restoration pipeline for improving generative audio quality
"""

__version__ = "1.0.0"
__author__ = "Alessandro"

from .restoration import AudioRestorer
from .metrics import AudioQualityMetrics

__all__ = ['AudioRestorer', 'AudioQualityMetrics']
