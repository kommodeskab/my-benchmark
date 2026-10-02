"""Audio Representation Learning Benchmark Suite."""

from audio_benchmark.bayes_opt import BayesOptClassifier, BayesOptRegressor
from audio_benchmark.benchmark import AudioBenchmark, BenchmarkResult
from audio_benchmark.cache import RepresentationCache
from audio_benchmark.classifiers import (
    BaseClassifier,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)
from audio_benchmark.dataset import (
    get_dataset_name,
    parse_dataset_item,
    validate_dataset,
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
    BaseTask,
    FrameClassificationTask,
    FrameRegressionTask,
    GlobalClassificationTask,
    RegressionTask,
    SequenceModelingTask,
    compute_token_error_rate,
    downsample_sequence,
    levenshtein_distance,
)
from audio_benchmark.utils import resolve_device, to_numpy
from audio_benchmark.wrappers import (
    BaseMetadataWrapper,
    PhonemeWrapper,
    VoiceActivityWrapper,
    WhisperTranscriptionWrapper,
)

__all__ = [
    "AudioBenchmark",
    "BaseClassifier",
    "BaseMetadataWrapper",
    "BaseRegressor",
    "BaseTask",
    "BayesOptClassifier",
    "BayesOptRegressor",
    "BenchmarkError",
    "BenchmarkResult",
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
    "PhonemeWrapper",
    "RegressionTask",
    "RepresentationCache",
    "SequenceModelingTask",
    "TaskError",
    "TransformerBenchmarkClassifier",
    "TransformerBenchmarkRegressor",
    "VoiceActivityWrapper",
    "WhisperTranscriptionWrapper",
    "compute_token_error_rate",
    "downsample_sequence",
    "extract_representations",
    "get_dataset_name",
    "levenshtein_distance",
    "parse_dataset_item",
    "resolve_device",
    "to_numpy",
    "validate_dataset",
]
