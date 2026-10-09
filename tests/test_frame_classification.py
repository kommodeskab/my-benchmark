"""Tests for FrameClassificationTask."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
)
from audio_benchmark.exceptions import InvalidDatasetError
from audio_benchmark.tasks.frame_classification import FrameClassificationTask


class MismatchedFrameDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        # Audio length 100 but frame_labels length 50
        return {
            "waveform": torch.randn(1, 100),
            "sample_rate": 16000,
            "metadata": {"frame_labels": [0] * 50},
        }


def test_frame_classification_with_knn_and_mlp():
    task_linear = FrameClassificationTask(classifier=LinearBenchmarkClassifier())
    task_knn = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    task_mlp = FrameClassificationTask(
        classifier=MLPBenchmarkClassifier(epochs=1, device="cpu"),
    )

    # 6 train clips, 2 test clips, each representation sequence length 4, dim 8
    train_reps = torch.randn(6, 4, 8)
    train_meta = [{"frame_labels": [i % 2] * 16} for i in range(6)]
    test_reps = torch.randn(2, 4, 8)
    test_meta = [{"frame_labels": [i % 2] * 16} for i in range(2)]

    info = task_knn.examine_data(train_meta, test_meta)
    assert info["train_clips"] == 6
    assert info["test_clips"] == 2
    assert info["frame_length"] == 16
    assert info["total_frames"] == 8 * 16
    assert info["train_frames"] == 6 * 16
    assert info["test_frames"] == 2 * 16
    assert info["baseline_strategy"] == "majority_class"
    assert info["baseline_metric"] == "accuracy"
    assert 0.0 <= info["baseline_value"] <= 1.0

    for task in (task_linear, task_knn, task_mlp):
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
        assert "train_clips" not in res
        assert "test_clips" not in res
        assert "train_frames" not in res
        assert "test_frames" not in res
        assert "T_orig" not in res
        assert "T_downsampled" not in res
        assert 0.0 <= res["accuracy"] <= 1.0
        assert "training_time" in res
        assert res["training_time"] >= 0.0


def test_frame_classification_dataset_validation():
    task = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), label_key="frame_labels"
    )
    bad_ds = MismatchedFrameDataset()

    with pytest.raises(InvalidDatasetError, match="Length mismatch in 'frame_labels'"):
        task.validate_dataset(bad_ds)


def test_benchmark_runner_with_frame_classification(tmp_path):
    train_dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    test_dataset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())

    benchmark = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task,
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=True,
    )

    result = benchmark.run(
        model=model,
        model_name="dummy_frame_clf_model",
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "frame_classification"
    assert "accuracy" in result.metrics
    assert "T_orig" not in result.metrics
    info = task.examine_data(train_dataset, test_dataset)
    assert info["frame_length"] == 320
    assert info["train_clips"] == 8
    assert info["test_clips"] == 4


def test_frame_classification_string_classifier_raises():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        FrameClassificationTask(classifier="knn")


def test_frame_classification_none_or_missing_classifier_raises():
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        FrameClassificationTask(classifier=None)
    with pytest.raises(TypeError):
        FrameClassificationTask()
