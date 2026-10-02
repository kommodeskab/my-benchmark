"""Comprehensive tests for task.examine_data() and baseline metric functionality."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    FrameClassificationTask,
    FrameRegressionTask,
    GlobalClassificationTask,
    KNNBenchmarkClassifier,
    LinearBenchmarkRegressor,
    RegressionTask,
    SequenceModelingTask,
)
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask


class SimpleAudioDataset(Dataset):
    """Simple test dataset."""

    def __init__(self, samples, audio_length: int = 160):
        self.samples = samples
        self.audio_length = audio_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, self.audio_length),
            "sample_rate": 16000,
            "metadata": self.samples[idx],
        }


def test_base_task_extract_metadata_sources():
    # 1. From list of dicts directly
    meta_list = [{"label": 0}, {"label": 1}]
    extracted = BaseTask.extract_metadata(meta_list)
    assert extracted == meta_list

    # 2. From Dataset object
    ds = SimpleAudioDataset(meta_list)
    extracted_ds = BaseTask.extract_metadata(ds)
    assert extracted_ds == meta_list

    # 3. From Dataset with .metadata attribute
    ds.metadata = [{"label": 2}, {"label": 3}]
    extracted_attr = BaseTask.extract_metadata(ds)
    assert extracted_attr == [{"label": 2}, {"label": 3}]


def test_global_classification_examine_data():
    # 10 samples: 7 class 'cat', 3 class 'dog'
    samples = [{"label": "cat"} for _ in range(7)] + [
        {"label": "dog"} for _ in range(3)
    ]
    ds = SimpleAudioDataset(samples)

    task = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(),
        test_size=0.3,
        random_state=42,
        metric="accuracy",
    )
    info = task.examine_data(ds)

    assert info["task"] == "global_classification"
    assert info["total_samples"] == 10
    assert info["train_samples"] == 7
    assert info["test_samples"] == 3
    assert info["num_classes"] == 2
    assert set(info["classes"]) == {"cat", "dog"}
    assert info["class_counts"] == {"cat": 7, "dog": 3}
    assert info["majority_class"] == "cat"
    assert info["baseline_strategy"] == "majority_class"
    assert info["baseline_metric"] == "accuracy"
    assert 0.0 <= info["baseline_value"] <= 1.0
    assert info["baseline_accuracy"] == info["baseline_value"]
    assert info["chance_level"] == 0.5


def test_global_classification_examine_data_custom_metric():
    samples = [{"label": f"c_{i % 3}"} for i in range(12)]
    ds = SimpleAudioDataset(samples)

    task = GlobalClassificationTask(
        classifier=KNNBenchmarkClassifier(), metric="macro_f1"
    )
    info = task.examine_data(ds)

    assert info["baseline_metric"] == "macro_f1"
    assert info["baseline_value"] == info["baseline_macro_f1"]
    assert info["num_classes"] == 3
    assert info["chance_level"] == pytest.approx(1.0 / 3.0)


def test_global_classification_examine_data_error_cases():
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    # Single class dataset
    single_class_ds = SimpleAudioDataset([{"label": "a"} for _ in range(5)])
    with pytest.raises(TaskError, match="requires at least 2 distinct classes"):
        task.examine_data(single_class_ds)

    # Missing label key
    missing_key_ds = SimpleAudioDataset([{"wrong": 1} for _ in range(5)])
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        task.examine_data(missing_key_ds)


def test_regression_examine_data():
    targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    samples = [{"target": t} for t in targets]
    ds = SimpleAudioDataset(samples)

    task = RegressionTask(
        regressor=LinearBenchmarkRegressor(),
        test_size=0.25,
        random_state=42,
        metric="r2",
    )
    info = task.examine_data(ds)

    assert info["task"] == "regression"
    assert info["total_samples"] == 8
    assert info["train_samples"] == 6
    assert info["test_samples"] == 2
    assert info["target_key"] == "target"
    assert info["target_min"] == 1.0
    assert info["target_max"] == 8.0
    assert info["target_mean"] == 4.5
    assert info["baseline_strategy"] == "mean_predictor"
    assert info["baseline_metric"] == "r2"
    assert "baseline_value" in info
    assert "baseline_mse" in info
    assert "baseline_rmse" in info
    assert "baseline_mae" in info


def test_regression_examine_data_custom_metric():
    samples = [{"target": float(i)} for i in range(10)]
    ds = SimpleAudioDataset(samples)

    task = RegressionTask(regressor=LinearBenchmarkRegressor(), metric="mse")
    info = task.examine_data(ds)

    assert info["baseline_metric"] == "mse"
    assert info["baseline_value"] == info["baseline_mse"]


def test_regression_examine_data_invalid_target():
    task = RegressionTask(regressor=LinearBenchmarkRegressor())
    bad_ds = SimpleAudioDataset([{"target": "invalid_number"} for _ in range(5)])
    with pytest.raises(TaskError, match="cannot be converted to float"):
        task.examine_data(bad_ds)


def test_frame_classification_examine_data():
    # 8 clips, each 16 frames
    samples = [
        {"frame_labels": [0 if j < 10 else 1 for j in range(16)]} for _ in range(8)
    ]
    ds = SimpleAudioDataset(samples, audio_length=16)

    task = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), test_size=0.25, random_state=42
    )
    info = task.examine_data(ds)

    assert info["task"] == "frame_classification"
    assert info["total_clips"] == 8
    assert info["train_clips"] == 6
    assert info["test_clips"] == 2
    assert info["frame_length"] == 16
    assert info["total_frames"] == 8 * 16
    assert info["train_frames"] == 6 * 16
    assert info["test_frames"] == 2 * 16
    assert info["num_classes"] == 2
    assert info["majority_class"] == "0"
    assert info["baseline_strategy"] == "majority_class"
    assert info["baseline_metric"] == "accuracy"
    assert 0.0 <= info["baseline_value"] <= 1.0


def test_frame_regression_examine_data():
    samples = [
        {"frame_targets": [float(i + j * 0.1) for j in range(10)]} for i in range(6)
    ]
    ds = SimpleAudioDataset(samples, audio_length=10)

    task = FrameRegressionTask(
        regressor=LinearBenchmarkRegressor(),
        test_size=0.33,
        random_state=42,
        metric="mae",
    )
    info = task.examine_data(ds)

    assert info["task"] == "frame_regression"
    assert info["total_clips"] == 6
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2
    assert info["frame_length"] == 10
    assert info["total_frames"] == 60
    assert info["train_frames"] == 40
    assert info["test_frames"] == 20
    assert info["baseline_strategy"] == "mean_predictor"
    assert info["baseline_metric"] == "mae"
    assert info["baseline_value"] == info["baseline_mae"]


def test_sequence_modeling_examine_data():
    samples = [
        {"tokens": ["hello", "world"]},
        {"tokens": ["foo", "bar", "baz"]},
        {"tokens": ["hello", "foo"]},
        {"tokens": ["world"]},
    ]
    ds = SimpleAudioDataset(samples)

    task = SequenceModelingTask(test_size=0.5, random_state=42)
    info = task.examine_data(ds)

    assert info["task"] == "sequence_modeling"
    assert info["total_clips"] == 4
    assert info["train_clips"] == 2
    assert info["test_clips"] == 2
    assert info["vocab_size"] == 5
    assert set(info["vocabulary"]) == {"hello", "world", "foo", "bar", "baz"}
    assert info["min_sequence_length"] == 1
    assert info["max_sequence_length"] == 3
    assert info["baseline_strategy"] == "empty_sequence"
    assert info["baseline_metric"] == "token_error_rate"
    assert info["baseline_value"] == 1.0
    assert info["baseline_token_error_rate"] == 1.0
    assert info["baseline_sequence_accuracy"] == 0.0


def test_audio_benchmark_end_to_end_extra_info(tmp_path, capsys):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyAudioDataset(num_samples=10, audio_length=320, num_classes=3)
    model = DummyAudioModel(representation_dim=16)
    task = GlobalClassificationTask(classifier=KNNBenchmarkClassifier())

    result = benchmark.run(
        model=model,
        model_name="test_model",
        dataset=dataset,
        task=task,
    )

    # Extra info populated from examine_data
    assert "train_samples" in result.extra_info
    assert "test_samples" in result.extra_info
    assert "baseline_value" in result.extra_info
    assert "baseline_strategy" in result.extra_info
    assert result.extra_info["baseline_strategy"] == "majority_class"
    assert result.extra_info["num_classes"] == 3

    # evaluate metrics should NOT contain dataset stats
    assert "train_samples" not in result.metrics
    assert "test_samples" not in result.metrics
    assert "num_classes" not in result.metrics
    assert "classes" not in result.metrics

    # print_summary prints cleanly
    result.print_summary()
    captured = capsys.readouterr().out
    assert "Baseline (accuracy) (majority_class):" in captured
    assert "Num Classes:     3" in captured
    assert "Split (Tr/Te):" in captured
