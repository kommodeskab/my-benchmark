"""Dataset wrappers for dynamic metadata generation."""

from audio_benchmark.wrappers.base import BaseMetadataWrapper
from audio_benchmark.wrappers.phonemes import PhonemeWrapper
from audio_benchmark.wrappers.voice_activity import VoiceActivityWrapper
from audio_benchmark.wrappers.whisper import WhisperTranscriptionWrapper

__all__ = [
    "BaseMetadataWrapper",
    "PhonemeWrapper",
    "VoiceActivityWrapper",
    "WhisperTranscriptionWrapper",
]
