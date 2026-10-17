"""Audio Restoration for MusicGen.

A zero-shot restoration pipeline for improving generative audio quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"
__author__ = "Alessandro Lo Curcio"

if TYPE_CHECKING:  # pragma: no cover - imported by type checkers only
    from .metrics import AudioQualityMetrics
    from .restoration import (
        PRESETS,
        AggressiveAudioRestorer,
        AudioRestorer,
        RestorationConfig,
        write_report,
    )

__all__ = [
    "AudioRestorer",
    "AggressiveAudioRestorer",
    "RestorationConfig",
    "PRESETS",
    "AudioQualityMetrics",
    "write_report",
]


def __getattr__(name: str) -> Any:
    """Import public objects lazily.

    Keeping the package initializer light avoids the ``runpy`` warning when a
    module is executed with ``python -m src.restoration`` and lets users import
    only what they need.
    """
    if name == "AudioQualityMetrics":
        from .metrics import AudioQualityMetrics
        return AudioQualityMetrics
    if name in {"AudioRestorer", "AggressiveAudioRestorer", "RestorationConfig",
                "PRESETS", "write_report"}:
        from .restoration import (
            PRESETS,
            AggressiveAudioRestorer,
            AudioRestorer,
            RestorationConfig,
            write_report,
        )
        return {
            "AudioRestorer": AudioRestorer,
            "AggressiveAudioRestorer": AggressiveAudioRestorer,
            "RestorationConfig": RestorationConfig,
            "PRESETS": PRESETS,
            "write_report": write_report,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
