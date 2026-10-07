"""End-to-end integration tests for AudioBenchmark."""

import pytest

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import KNNBenchmarkClassifier, MLPBenchmarkClassifier
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.tasks.global_classification import GlobalClassificationTask


def test_benchmark_end_to_end_dummy(tmp_path):
    # Tiny dataset and short audio length for near-instant execution
    train_dataset = DummyAudioDataset(num_samples=12, audio_length=320, num_classes=3)
    test_dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=3)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        model=model,
        model_name="dummy_encoder_v1",
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        cache_dir=tmp_path,
        enable_cache=True,
    )

    # 1. First run: should compute and cache representations
    result1 = benchmark.run(
        batch_size=4,
        device="cpu",
    )

    assert result1.model_name == "dummy_encoder_v1"
    assert not result1.cached
    assert result1.representations_shape[0] == 12
    assert "accuracy" in result1.metrics

    # 2. Second run: should hit cache
    result2 = benchmark.run(
        batch_size=4,
        device="cpu",
    )

    assert result2.cached
    assert result2.metrics["accuracy"] == result1.metrics["accuracy"]

    # 3. Third run with force_recompute=True: should bypass cache
    result3 = benchmark.run(
        batch_size=4,
        device="cpu",
        force_recompute=True,
    )

    assert not result3.cached


def test_benchmark_requires_testset(tmp_path):
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    with pytest.raises(TypeError):
        AudioBenchmark(
            model=model,
            model_name="dummy_model",
            dataset=dataset,
            task=task,
            cache_dir=tmp_path,
            enable_cache=False,
        )

    with pytest.raises(TypeError, match="testset must be an instantiated Dataset"):
        AudioBenchmark(
            model=model,
            model_name="dummy_model",
            dataset=dataset,
            testset=None,
            task=task,
            cache_dir=tmp_path,
            enable_cache=False,
        )


def test_benchmark_requires_base_task(tmp_path):
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # Passing string instead of BaseTask must raise TypeError
    with pytest.raises(TypeError, match="task must be an instance of BaseTask"):
        AudioBenchmark(
            model=model,
            model_name="dummy_model",
            dataset=dataset,
            testset=testset,
            task="global_classification",
            cache_dir=tmp_path,
            enable_cache=False,
        )


def test_benchmark_with_custom_task_instance(tmp_path):
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    task = GlobalClassificationTask(
        classifier=MLPBenchmarkClassifier(epochs=1, device="cpu"),
    )

    benchmark = AudioBenchmark(
        model=model,
        model_name="dummy_mlp_test",
        dataset=dataset,
        testset=testset,
        task=task,
        cache_dir=tmp_path,
        enable_cache=False,
    )

    result = benchmark.run(
        batch_size=5,
        device="cpu",
    )

    assert result.metrics["classifier"] == "MLPBenchmarkClassifier"
    assert "accuracy" in result.metrics


def test_benchmark_show_progress_in_run(tmp_path):
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        model=model,
        model_name="dummy_progress_test",
        dataset=dataset,
        testset=testset,
        task=task,
        cache_dir=tmp_path,
        enable_cache=False,
    )
    assert not hasattr(benchmark, "show_progress")

    result = benchmark.run(
        batch_size=4,
        device="cpu",
        show_progress=False,
    )
    assert "accuracy" in result.metrics


def test_benchmark_run_rejects_benchmark_config_args(tmp_path):
    dataset = DummyAudioDataset(num_samples=6, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        model=model,
        model_name="dummy_model",
        dataset=dataset,
        testset=testset,
        task=task,
        cache_dir=tmp_path,
        enable_cache=False,
    )

    with pytest.raises(TypeError):
        benchmark.run(model=model)
