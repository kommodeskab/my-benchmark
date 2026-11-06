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

    # Valid flat tensor targets dataset
    class TensorFrameDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_labels": torch.tensor([0, 1])},
            }

    task.validate_dataset(TensorFrameDataset())

    # Invalid cases: dictionary targets are rejected
    class DictFrameDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "waveform": torch.randn(1, 100),
                "sample_rate": 16000,
                "metadata": {"frame_labels": {0: 0, 50: 1}},
            }

    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        task.validate_dataset(DictFrameDataset())

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


def test_frame_classification_variable_length_and_tensor():
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())

    # Clips with varying target counts: clip 0 has 3 targets, clip 1 has 5 targets, clip 2 has flat tensor
    train_reps = torch.randn(4, 6, 8)
    train_meta = [
        {"frame_labels": [0, 1, 0], "audio_length": 60},  # K=3 (downsample 6 -> 3)
        {
            "frame_labels": [1, 0, 1, 0, 1],
            "audio_length": 60,
        },  # K=5 (downsample 6 -> 5)
        {
            "frame_labels": torch.tensor([0, 1]),
            "audio_length": 60,
        },  # flat tensor K=2 (downsample 6 -> 2)
        {"frame_labels": [0] * 12, "audio_length": 60},  # K=12 (upsample 6 -> 12)
    ]
    test_reps = torch.randn(2, 6, 8)
    test_meta = [
        {"frame_labels": torch.tensor([1, 0]), "audio_length": 60},  # flat tensor K=2
        {"frame_labels": [1, 1, 0, 0], "audio_length": 60},  # K=4
    ]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert "training_time" in res

    # Verify examine_data computes correct stats on variable length / tensor targets
    info = task.examine_data(
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


def test_frame_classification_rejects_dict():
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    with pytest.raises(
        InvalidDatasetError, match="must be a flat torch.Tensor or list"
    ):
        task.evaluate(
            torch.randn(1, 4, 8),
            [{"frame_labels": {0: 0, 10: 1}}],
            torch.randn(1, 4, 8),
            [{"frame_labels": [0, 1]}],
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

    results = benchmark.run(
        model=model,
        model_name="dummy_frame_clf_model",
        batch_size=4,
        device="cpu",
    )

    assert isinstance(results, list)
    assert len(results) == 2
    result = results[0]
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


def test_frame_classification_slices_to_rep_length():
    """Verify trailing padded frames beyond rep_length are ignored before interpolation."""
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier(n_neighbors=1))

    # Sample 0 has rep_length=2 (valid frames: [0.0], [10.0]; padded frames: [999.0, 999.0])
    # Sample 1 has rep_length=4 (valid frames: [0.0], [0.0], [10.0], [10.0])
    train_reps = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 10.0], [999.0, 999.0], [999.0, 999.0]],
            [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    train_meta = [
        {"frame_labels": [0, 1], "rep_length": 2},
        {"frame_labels": [0, 0, 1, 1], "rep_length": 4},
    ]

    # Test sample has rep_length=2 with huge corrupt values in the padded tail
    test_reps = torch.tensor(
        [[[0.0, 0.0], [10.0, 10.0], [-999.0, -999.0], [999.0, 999.0]]],
        dtype=torch.float32,
    )
    test_meta = [{"frame_labels": [0, 1], "rep_length": 2}]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["accuracy"] == 1.0


def test_frame_classification_ignore_label():
    """Verify ignore_label excludes specified frames from training, evaluation, and examine_data."""
    task = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(n_neighbors=1),
        ignore_label=0,
    )

    # Frame 0 is class 0 (ignored with noisy features), Frame 1 is class 1, Frame 2 is class 2
    train_reps = torch.tensor(
        [
            [[999.0, -999.0], [1.0, 1.0], [10.0, 10.0]],
            [[-888.0, 888.0], [1.0, 1.0], [10.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    train_meta = [
        {"frame_labels": [0, 1, 2]},
        {"frame_labels": [0, 1, 2]},
    ]
    test_reps = torch.tensor(
        [[[555.0, 555.0], [1.0, 1.0], [10.0, 10.0]]],
        dtype=torch.float32,
    )
    test_meta = [{"frame_labels": [0, 1, 2]}]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["accuracy"] == 1.0

    info = task.examine_data(
        SimpleDataset(train_meta, audio_length=30),
        SimpleDataset(test_meta, audio_length=30),
    )
    assert info["num_classes"] == 2
    assert "0" not in info["classes"]
    assert info["train_frames"] == 4
    assert info["test_frames"] == 2
