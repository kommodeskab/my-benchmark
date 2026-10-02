"""Tests for FrameClassificationTask."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import AudioBenchmark, DummyAudioDataset, DummyAudioModel
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
    task_linear = FrameClassificationTask(
        classifier="linear", test_size=0.25, random_state=42
    )
    task_knn = FrameClassificationTask(
        classifier="knn", test_size=0.25, random_state=42
    )
    task_mlp = FrameClassificationTask(
        classifier="mlp",
        test_size=0.25,
        random_state=42,
        classifier_kwargs={"epochs": 1, "device": "cpu"},
    )

    # 8 clips, each representation sequence length 4, dim 8
    reps = torch.randn(8, 4, 8)
    # Original audio had 16 frames, downsampled to 4
    meta = [{"frame_labels": [i % 2] * 16} for i in range(8)]

    info = task_knn.examine_data(meta)
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
        res = task.evaluate(reps, meta)
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
    task = FrameClassificationTask(label_key="frame_labels")
    bad_ds = MismatchedFrameDataset()

    with pytest.raises(InvalidDatasetError, match="Length mismatch in 'frame_labels'"):
        task.validate_dataset(bad_ds)


def test_frame_classification_with_split_key():
    task = FrameClassificationTask(classifier="knn", split_key="split")
    reps = torch.randn(6, 3, 8)
    meta = []
    for i in range(6):
        split = "train" if i < 4 else "test"
        meta.append({"frame_labels": [i % 2] * 12, "split": split})

    info = task.examine_data(meta)
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2

    res = task.evaluate(reps, meta)
    assert "train_clips" not in res
    assert "train_frames" not in res


def test_benchmark_runner_with_frame_classification(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=True)
    # DummyAudioDataset generates frame_labels of length audio_length
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)
    task = FrameClassificationTask(classifier="knn")

    result = benchmark.run(
        model=model,
        model_name="dummy_frame_clf_model",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert result.task_name == "frame_classification"
    assert "accuracy" in result.metrics
    assert "T_orig" not in result.metrics
    assert result.extra_info["frame_length"] == 320
    assert result.extra_info["train_clips"] == 6
