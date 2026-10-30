"""Pre-defined audio benchmarks package."""

from audio_benchmark.benchmarks.cremad import CREMADClassificationBenchmark
from audio_benchmark.benchmarks.ptdb import (
    PTDBPitchClassificationBenchmark,
    PTDBVoiceActivityClassificationBenchmark,
)
from audio_benchmark.benchmarks.speech_commands import (
    SpeechCommandsClassificationBenchmark,
)
from audio_benchmark.benchmarks.timit import (
    TIMITSpeechDurationRegressionBenchmark,
)
from audio_benchmark.benchmarks.timit_phonemes import (
    TIMITPhonemeSequenceModelingBenchmark,
)
from audio_benchmark.benchmarks.timit_text import (
    TIMITTextSequenceModelingBenchmark,
)
from audio_benchmark.benchmarks.vctk import VCTKGenderClassificationBenchmark

__all__ = [
    "CREMADClassificationBenchmark",
    "PTDBPitchClassificationBenchmark",
    "PTDBVoiceActivityClassificationBenchmark",
    "SpeechCommandsClassificationBenchmark",
    "TIMITPhonemeSequenceModelingBenchmark",
    "TIMITSpeechDurationRegressionBenchmark",
    "TIMITTextSequenceModelingBenchmark",
    "VCTKGenderClassificationBenchmark",
]
