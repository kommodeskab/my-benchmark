"""Audio Representation Learning Benchmark Suite."""

from audio_benchmark.benchmark import AudioBenchmark, BenchmarkResult
from audio_benchmark.benchmarks import (
    CREMADClassificationBenchmark,
    SpeechCommandsClassificationBenchmark,
    TIMITPhonemeSequenceModelingBenchmark,
    TIMITSpeechDurationRegressionBenchmark,
    TIMITTextSequenceModelingBenchmark,
    VCTKGenderClassificationBenchmark,
)
from audio_benchmark.cache import RepresentationCache
from audio_benchmark.classifiers import (
    BaseClassifier,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)
from audio_benchmark.dataset import (
    BaseDataset,
    deterministic_random_subset,
    parse_dataset_item,
    validate_dataset,
)
from audio_benchmark.datasets import (
    CREMADDataset,
    PTDBTUGDataset,
    SpeechCommandsDataset,
    TIMITDataset,
    VCTKDataset,
)
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.exceptions import (
    BenchmarkError,
    InvalidDatasetError,
    InvalidModelError,
    InvalidModelOutputError,
    TaskError,
)
from audio_benchmark.extractor import extract_representations
from audio_benchmark.regressors import (
    BaseRegressor,
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)
from audio_benchmark.tasks import (
    BaseSequenceProbe,
    BaseTask,
    CTCSequenceProbe,
    FrameClassificationTask,
    FrameRegressionTask,
    GlobalClassificationTask,
    RegressionTask,
    SequenceModelingTask,
    TransformerDecoderProbe,
    compute_token_error_rate,
    levenshtein_distance,
)
from audio_benchmark.utils import resolve_device

__all__ = [
    "AudioBenchmark",
    "BaseClassifier",
    "BaseDataset",
    "BaseRegressor",
    "BaseSequenceProbe",
    "BaseTask",
    "BenchmarkError",
    "BenchmarkResult",
    "CREMADClassificationBenchmark",
    "CREMADDataset",
    "CTCSequenceProbe",
    "DummyAudioDataset",
    "DummyAudioModel",
    "FrameClassificationTask",
    "FrameRegressionTask",
    "GlobalClassificationTask",
    "InvalidDatasetError",
    "InvalidModelError",
    "InvalidModelOutputError",
    "KNNBenchmarkClassifier",
    "KNNBenchmarkRegressor",
    "LinearBenchmarkClassifier",
    "LinearBenchmarkRegressor",
    "MLPBenchmarkClassifier",
    "MLPBenchmarkRegressor",
    "PTDBTUGDataset",
    "RegressionTask",
    "RepresentationCache",
    "SequenceModelingTask",
    "SpeechCommandsClassificationBenchmark",
    "SpeechCommandsDataset",
    "TIMITDataset",
    "TIMITPhonemeSequenceModelingBenchmark",
    "TIMITSpeechDurationRegressionBenchmark",
    "TIMITTextSequenceModelingBenchmark",
    "TaskError",
    "TransformerBenchmarkClassifier",
    "TransformerBenchmarkRegressor",
    "TransformerDecoderProbe",
    "VCTKDataset",
    "VCTKGenderClassificationBenchmark",
    "compute_token_error_rate",
    "deterministic_random_subset",
    "extract_representations",
    "levenshtein_distance",
    "parse_dataset_item",
    "resolve_device",
    "validate_dataset",
]
