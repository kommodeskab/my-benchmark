from audio_benchmark.regressors.models import (
    BaseRegressor,
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)

__all__ = [
    "BaseRegressor",
    "BayesOptRegressor",
    "KNNBenchmarkRegressor",
    "LinearBenchmarkRegressor",
    "MLPBenchmarkRegressor",
    "TransformerBenchmarkRegressor",
]


def __getattr__(name: str):
    if name == "BayesOptRegressor":
        from audio_benchmark.bayes_opt import BayesOptRegressor

        return BayesOptRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
