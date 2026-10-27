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


class SimpleDataset(Dataset):
    def __init__(self, samples, audio_length: int = 16):
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

    info = task_knn.examine_data(
        SimpleDataset(train_meta, audio_length=16),
        SimpleDataset(test_meta, audio_length=16),
    )
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
    # Arbitrary length 1D sequence (e.g. 50 elements for 100 audio samples) is now accepted
    valid_seq_ds = MismatchedFrameDataset()
    task.validate_dataset(valid_seq_ds)

    # Valid dictionary targets dataset
    class DictFrameDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_labels": {0: 0, 50: 1}},
            }

    task.validate_dataset(DictFrameDataset())

    # Invalid cases: out of bounds timestamp >= audio_length
    class OutOfBoundsFrameDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_labels": {100: 0}},
            }

    with pytest.raises(InvalidDatasetError, match="exceeds audio length"):
        task.validate_dataset(OutOfBoundsFrameDataset())

    # Invalid cases: empty sequence
    class EmptyFrameDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_labels": []},
            }

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        task.validate_dataset(EmptyFrameDataset())


def test_frame_classification_variable_length_and_dictionary():
    task_mean = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), pooling="mean"
    )
    task_max = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), pooling="max"
    )

    # Clips with varying target counts: clip 0 has 3 targets, clip 1 has 5 targets, clip 2 has dict
    train_reps = torch.randn(4, 6, 8)
    train_meta = [
        {"frame_labels": [0, 1, 0], "audio_length": 60},  # K=3 (downsample 6 -> 3)
        {
            "frame_labels": [1, 0, 1, 0, 1],
            "audio_length": 60,
        },  # K=5 (downsample 6 -> 5)
        {
            "frame_labels": {0: 0, 30: 1},
            "audio_length": 60,
        },  # dict K=2 (downsample 6 -> 2)
        {"frame_labels": [0] * 12, "audio_length": 60},  # K=12 (upsample 6 -> 12)
    ]
    test_reps = torch.randn(2, 6, 8)
    test_meta = [
        {"frame_labels": {10: 1, 40: 0}, "audio_length": 60},  # dict K=2
        {"frame_labels": [1, 1, 0, 0], "audio_length": 60},  # K=4
    ]

    for task in (task_mean, task_max):
        res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
        assert 0.0 <= res["accuracy"] <= 1.0
        assert "training_time" in res

    # Verify examine_data computes correct stats on variable length / dict targets
    info = task_mean.examine_data(
        SimpleDataset(train_meta, audio_length=60),
        SimpleDataset(test_meta, audio_length=60),
    )
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2
    # Total train frames: 3 + 5 + 2 + 12 = 22
    assert info["train_frames"] == 22
    # Total test frames: 2 + 4 = 6
    assert info["test_frames"] == 6
    assert info["total_frames"] == 28
    assert info["num_classes"] == 2


def test_frame_classification_invalid_pooling():
    with pytest.raises(ValueError, match="Unknown pooling"):
        FrameClassificationTask(
            classifier=KNNBenchmarkClassifier(), pooling="unsupported"
        )


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
    with pytest.raises(
        TypeError,
        match=r"classifier must be an instantiated BaseClassifier instance, got str",
    ):
        FrameClassificationTask(classifier="knn")


def test_frame_classification_none_or_missing_classifier_raises():
    with pytest.raises(
        TypeError,
        match=r"classifier must be an instantiated BaseClassifier instance, got NoneType",
    ):
        FrameClassificationTask(classifier=None)
    with pytest.raises(TypeError):
        FrameClassificationTask()
