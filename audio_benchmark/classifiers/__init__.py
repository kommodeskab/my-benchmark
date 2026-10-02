from audio_benchmark.classifiers.models import (
    BaseClassifier,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)

__all__ = [
    "BaseClassifier",
    "BayesOptClassifier",
    "KNNBenchmarkClassifier",
    "LinearBenchmarkClassifier",
    "MLPBenchmarkClassifier",
    "TransformerBenchmarkClassifier",
]


def __getattr__(name: str):
    if name == "BayesOptClassifier":
        from audio_benchmark.bayes_opt import BayesOptClassifier

        return BayesOptClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
