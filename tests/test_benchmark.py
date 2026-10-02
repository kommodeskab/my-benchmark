"""End-to-end integration tests for AudioBenchmark."""

import pytest

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import KNNBenchmarkClassifier, MLPBenchmarkClassifier
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.tasks.global_classification import GlobalClassificationTask


def test_benchmark_end_to_end_dummy(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=True)
    # Tiny dataset and short audio length for near-instant execution
    dataset = DummyAudioDataset(num_samples=12, audio_length=320, num_classes=3)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    # 1. First run: should compute and cache representations
    result1 = benchmark.run(
        model=model,
        model_name="dummy_encoder_v1",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert result1.model_name == "dummy_encoder_v1"
    assert not result1.cached
    assert result1.representations_shape[0] == 12
    assert "accuracy" in result1.metrics

    # 2. Second run: should hit cache
    result2 = benchmark.run(
        model=model,
        model_name="dummy_encoder_v1",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert result2.cached
    assert result2.metrics["accuracy"] == result1.metrics["accuracy"]

    # 3. Third run with force_recompute=True: should bypass cache
    result3 = benchmark.run(
        model=model,
        model_name="dummy_encoder_v1",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
        force_recompute=True,
    )

    assert not result3.cached


def test_benchmark_requires_base_task(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # Passing string instead of BaseTask must raise TypeError
    with pytest.raises(TypeError, match="task must be an instance of BaseTask"):
        benchmark.run(
            model=model,
            model_name="dummy_model",
            dataset=dataset,
            task="global_classification",
        )


def test_benchmark_with_custom_task_instance(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    task = GlobalClassificationTask(
        classifier=MLPBenchmarkClassifier(epochs=1, device="cpu"),
        test_size=0.3,
    )

    result = benchmark.run(
        model=model,
        model_name="dummy_mlp_test",
        dataset=dataset,
        task=task,
        batch_size=5,
        device="cpu",
    )

    assert result.metrics["classifier"] == "MLPBenchmarkClassifier"
    assert "accuracy" in result.metrics
