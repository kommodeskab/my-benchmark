"""Datasets module for audio benchmark."""

from audio_benchmark.datasets.cremad import CREMADDataset
from audio_benchmark.datasets.ptdb import PTDBTUGDataset
from audio_benchmark.datasets.speechcommands import (
    SPEECHCOMMANDS_LABELS,
    SpeechCommandsDataset,
)
from audio_benchmark.datasets.timit import TIMITDataset
from audio_benchmark.datasets.vctk import VCTKDataset

__all__ = [
    "SPEECHCOMMANDS_LABELS",
    "CREMADDataset",
    "PTDBTUGDataset",
    "SpeechCommandsDataset",
    "TIMITDataset",
    "VCTKDataset",
]
