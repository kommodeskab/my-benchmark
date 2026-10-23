"""Adversarial audit tests for arbitrary-length and dictionary-mapped temporal alignment."""

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import KNNBenchmarkClassifier
from audio_benchmark.dummy import DummyAudioModel
from audio_benchmark.exceptions import InvalidDatasetError
from audio_benchmark.regressors import LinearBenchmarkRegressor
from audio_benchmark.tasks.downsampling import (
    align_features_to_targets,
    downsample_sequence,
    parse_frame_targets,
)
from audio_benchmark.tasks.frame_classification import FrameClassificationTask
from audio_benchmark.tasks.frame_regression import FrameRegressionTask


def test_align_features_mathematical_downsampling_exact_formulas():
    audio_length = 120.0
    # T' = 6 representation frames at centers (j + 0.5) * 120 / 6 = [10, 30, 50, 70, 90, 110]
    # K = 3 target frames at centers (i + 0.5) * 120 / 3 = [20, 60, 100]
    # Nearest mapping:
    # reps 0, 1 (10, 30) -> target 0 (20)
    # reps 2, 3 (50, 70) -> target 1 (60)
    # reps 4, 5 (90, 110) -> target 2 (100)
    reps = np.array(
        [
            [1.0, 10.0, 100.0],
            [3.0, 20.0, 200.0],
            [5.0, 30.0, 300.0],
            [7.0, 40.0, 400.0],
            [11.0, 50.0, 500.0],
            [13.0, 60.0, 600.0],
        ],
        dtype=np.float32,
    )
    target_ts = np.array([20.0, 60.0, 100.0], dtype=np.float64)

    # 1. Mean pooling
    mean_out = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="mean"
    )
    assert mean_out.shape == (3, 3)
    assert np.allclose(mean_out[0], [2.0, 15.0, 150.0])
    assert np.allclose(mean_out[1], [6.0, 35.0, 350.0])
    assert np.allclose(mean_out[2], [12.0, 55.0, 550.0])

    # 2. Max pooling
    max_out = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="max"
    )
    assert np.allclose(max_out[0], [3.0, 20.0, 200.0])
    assert np.allclose(max_out[1], [7.0, 40.0, 400.0])
    assert np.allclose(max_out[2], [13.0, 60.0, 600.0])

    # 3. Min pooling
    min_out = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="min"
    )
    assert np.allclose(min_out[0], [1.0, 10.0, 100.0])
    assert np.allclose(min_out[1], [5.0, 30.0, 300.0])
    assert np.allclose(min_out[2], [11.0, 50.0, 500.0])

    # 4. Median pooling (even number of elements: average of 2 elements)
    median_out = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="median"
    )
    assert np.allclose(median_out[0], [2.0, 15.0, 150.0])
    assert np.allclose(median_out[1], [6.0, 35.0, 350.0])
    assert np.allclose(median_out[2], [12.0, 55.0, 550.0])


def test_align_features_median_odd_pooling():
    audio_length = 90.0
    # T' = 6 representation frames at centers [7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
    # K = 2 target frames at centers [22.5, 67.5]
    # Target 0 gets reps [0, 1, 2] (3 frames, odd count)
    # Target 1 gets reps [3, 4, 5] (3 frames, odd count)
    reps = np.array(
        [
            [10.0, 50.0],
            [1.0, 20.0],
            [8.0, 90.0],
            [2.0, 30.0],
            [40.0, 5.0],
            [15.0, 80.0],
        ],
        dtype=np.float32,
    )
    target_ts = np.array([22.5, 67.5], dtype=np.float64)

    out = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="median"
    )
    assert out.shape == (2, 2)
    # Target 0: col 0 has {10, 1, 8} -> median is 8.0; col 1 has {50, 20, 90} -> median is 50.0
    assert np.allclose(out[0], [8.0, 50.0])
    # Target 1: col 0 has {2, 40, 15} -> median is 15.0; col 1 has {30, 5, 80} -> median is 30.0
    assert np.allclose(out[1], [15.0, 30.0])


def test_align_features_mathematical_upsampling_exact_formulas():
    audio_length = 120.0
    # T' = 3 representation frames at [20.0, 60.0, 100.0]
    # K = 6 target frames at [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
    # Target 0 (10.0) -> rep 0 (20.0)
    # Target 1 (30.0) -> rep 0 (20.0)
    # Target 2 (50.0) -> rep 1 (60.0)
    # Target 3 (70.0) -> rep 1 (60.0)
    # Target 4 (90.0) -> rep 2 (100.0)
    # Target 5 (110.0) -> rep 2 (100.0)
    reps = np.array(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [100.0, 200.0],
        ],
        dtype=np.float32,
    )
    target_ts = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 110.0], dtype=np.float64)

    aligned = align_features_to_targets(reps, target_ts, audio_length=audio_length)
    assert aligned.shape == (6, 2)
    assert np.allclose(aligned[0], [1.0, 2.0])
    assert np.allclose(aligned[1], [1.0, 2.0])
    assert np.allclose(aligned[2], [10.0, 20.0])
    assert np.allclose(aligned[3], [10.0, 20.0])
    assert np.allclose(aligned[4], [100.0, 200.0])
    assert np.allclose(aligned[5], [100.0, 200.0])


def test_align_features_boundaries_tprime_1_and_k_1():
    # Case 1: T' = 1, K = 1 (identity single frame)
    reps_single = np.array([[3.14, 2.71]], dtype=np.float32)
    out1 = align_features_to_targets(
        reps_single, np.array([25.0]), audio_length=50.0, pooling="mean"
    )
    assert out1.shape == (1, 2)
    assert np.allclose(out1[0], [3.14, 2.71])

    # Case 2: T' = 1, K = 4 (upsampling repetition of single rep frame)
    out2 = align_features_to_targets(
        reps_single,
        np.array([5.0, 15.0, 35.0, 45.0]),
        audio_length=50.0,
    )
    assert out2.shape == (4, 2)
    for row in out2:
        assert np.allclose(row, [3.14, 2.71])

    # Case 3: T' = 4, K = 1 (downsampling pooling all 4 frames)
    reps_four = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    out_mean = align_features_to_targets(
        reps_four, np.array([25.0]), audio_length=50.0, pooling="mean"
    )
    assert np.allclose(out_mean[0], [2.5])
    out_max = align_features_to_targets(
        reps_four, np.array([25.0]), audio_length=50.0, pooling="max"
    )
    assert np.allclose(out_max[0], [4.0])
    out_min = align_features_to_targets(
        reps_four, np.array([25.0]), audio_length=50.0, pooling="min"
    )
    assert np.allclose(out_min[0], [1.0])
    out_median = align_features_to_targets(
        reps_four, np.array([25.0]), audio_length=50.0, pooling="median"
    )
    assert np.allclose(out_median[0], [2.5])


def test_align_features_zero_assigned_target_fallback_multiple():
    audio_length = 100.0
    # T' = 8 reps at [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75]
    # K = 5 targets at [18.0, 18.1, 18.2, 81.0, 81.1]
    # Rep nearest targets:
    # rep 0 (6.25) -> target 0 (18.0)
    # rep 1 (18.75) -> target 2 (18.2)
    # rep 2 (31.25) -> target 2 (18.2)
    # rep 3 (43.75) -> target 2 (18.2)
    # rep 4 (56.25) -> target 3 (81.0)
    # rep 5 (68.75) -> target 3 (81.0)
    # rep 6 (81.25) -> target 4 (81.1)
    # rep 7 (93.75) -> target 4 (81.1)
    # Target 0: assigned rep 0
    # Target 1 (18.1): 0 assigned reps -> nearest rep is rep 1 (18.75, dist 0.65)
    # Target 2: assigned reps 1, 2, 3 -> pooled
    # Target 3: assigned reps 4, 5 -> pooled
    # Target 4: assigned reps 6, 7 -> pooled
    reps = np.array(
        [
            [10.0],
            [20.0],
            [30.0],
            [40.0],
            [100.0],
            [200.0],
            [1000.0],
            [2000.0],
        ],
        dtype=np.float32,
    )
    target_ts = np.array([18.0, 18.1, 18.2, 81.0, 81.1], dtype=np.float64)

    aligned = align_features_to_targets(
        reps, target_ts, audio_length=audio_length, pooling="mean"
    )
    assert aligned.shape == (5, 1)
    # Target 0: rep 0: 10.0
    assert np.allclose(aligned[0], [10.0])
    # Target 1: fallback to nearest rep 1: 20.0
    assert np.allclose(aligned[1], [20.0])
    # Target 2: mean of reps 1, 2, 3: (20 + 30 + 40) / 3 = 30.0
    assert np.allclose(aligned[2], [30.0])
    # Target 3: mean of reps 4, 5: (100 + 200) / 2 = 150.0
    assert np.allclose(aligned[3], [150.0])
    # Target 4: mean of reps 6, 7: (1000 + 2000) / 2 = 1500.0
    assert np.allclose(aligned[4], [1500.0])


def test_align_features_dense_dictionary_keys():
    audio_length = 100.0
    reps = np.ones((5, 8), dtype=np.float32)
    # 50 dense target keys
    dense_keys = np.arange(0, 100, 2, dtype=np.float64)
    out = align_features_to_targets(
        reps, dense_keys, audio_length=audio_length, pooling="mean"
    )
    assert out.shape == (50, 8)


def test_parse_frame_targets_stringified_and_dense_keys():
    # Stringified keys with surrounding whitespace
    ts, vals = parse_frame_targets(
        {"  0 ": 10, "50": 20, " 99": 30},
        audio_length=100,
        key_name="test_labels",
    )
    assert list(ts) == [0.0, 50.0, 99.0]
    assert vals == [10, 20, 30]

    # Non-integer strings raise InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="not a valid integer"):
        parse_frame_targets({"10.5": 1}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="not a valid integer"):
        parse_frame_targets({"abc": 1}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="not a valid integer"):
        parse_frame_targets({"+10": 1}, audio_length=100)

    # Boolean key raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must be an integer, got bool"):
        parse_frame_targets({True: 1}, audio_length=100)

    # Negative keys raise InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must be non-negative"):
        parse_frame_targets({-5: 1}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="must be non-negative"):
        parse_frame_targets({"-10": 1}, audio_length=100)

    # Duplicate timestamp keys raise InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="Duplicate timestamp key"):
        parse_frame_targets({10: 1, "10": 2}, audio_length=100)

    # Key exceeding audio_length raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="exceeds audio length"):
        parse_frame_targets({100: 1}, audio_length=100)

    # Empty dictionary raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets({}, audio_length=100)

    # Empty sequence raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets([], audio_length=100)

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets(torch.empty(0), audio_length=100)

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets(np.array([]), audio_length=100)

    # Multi-dimensional sequence raises InvalidDatasetError
    with pytest.raises(InvalidDatasetError, match="must be 1D"):
        parse_frame_targets(torch.zeros(2, 3), audio_length=100)

    # Non-positive audio_length raises ValueError
    with pytest.raises(ValueError, match="audio_length must be a positive number"):
        parse_frame_targets({"10": 1}, audio_length=0)

    with pytest.raises(ValueError, match="audio_length must be a positive number"):
        parse_frame_targets([1, 2], audio_length=-10)


def test_strict_validation_negative_out_of_bounds_and_invalid_pooling():
    reps = np.ones((4, 2), dtype=np.float32)

    # Out of bounds target timestamps
    with pytest.raises(ValueError, match="within"):
        align_features_to_targets(reps, np.array([-1.0, 50.0]), audio_length=100.0)

    with pytest.raises(ValueError, match="within"):
        align_features_to_targets(reps, np.array([50.0, 100.0]), audio_length=100.0)

    # Invalid audio_length
    with pytest.raises(ValueError, match="audio_length must be positive"):
        align_features_to_targets(reps, np.array([50.0]), audio_length=0)

    with pytest.raises(ValueError, match="audio_length must be positive"):
        align_features_to_targets(reps, np.array([50.0]), audio_length=-5)

    with pytest.raises(ValueError, match="audio_length must be positive"):
        align_features_to_targets(reps, np.array([50.0]), audio_length=True)

    # Invalid pooling
    with pytest.raises(ValueError, match="Unknown pooling"):
        align_features_to_targets(
            reps, np.array([50.0]), audio_length=100.0, pooling="mode"
        )

    # Invalid representation shapes
    with pytest.raises(ValueError, match="must have shape"):
        align_features_to_targets(np.ones(4), np.array([50.0]), audio_length=100.0)

    with pytest.raises(ValueError, match="temporal length 0"):
        align_features_to_targets(
            np.empty((0, 2)), np.array([50.0]), audio_length=100.0
        )

    # Invalid target_len in downsample_sequence
    with pytest.raises(ValueError, match="target_len must be positive"):
        downsample_sequence([1, 2, 3], target_len=0)

    with pytest.raises(ValueError, match="target_len must be positive"):
        downsample_sequence([1, 2, 3], target_len=-1)


def test_mismatched_clip_durations_evaluation():
    # 3 clips with different audio durations and target counts:
    # Clip 0: audio_length 60, T' = 6, K = 3 (downsampling 6 -> 3)
    # Clip 1: audio_length 120, T' = 4, K = 8 (upsampling 4 -> 8)
    # Clip 2: audio_length 80, T' = 4, dict K = 2 (downsampling 4 -> 2)
    task_clf = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    task_reg = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    train_reps = [
        torch.randn(6, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
    ]
    train_meta_clf = [
        {"frame_labels": [0, 1, 0], "audio_length": 60},
        {"frame_labels": [1, 0, 1, 0, 1, 0, 1, 0], "audio_length": 120},
        {"frame_labels": {10: 0, 50: 1}, "audio_length": 80},
    ]
    test_reps = [torch.randn(4, 4)]
    test_meta_clf = [{"frame_labels": [0, 1, 0, 1], "audio_length": 80}]

    res_clf = task_clf.evaluate(train_reps, train_meta_clf, test_reps, test_meta_clf)
    assert 0.0 <= res_clf["accuracy"] <= 1.0

    train_meta_reg = [
        {"frame_targets": [1.0, 2.0, 3.0], "audio_length": 60},
        {"frame_targets": [float(i) for i in range(8)], "audio_length": 120},
        {"frame_targets": {10: 1.5, 50: 3.5}, "audio_length": 80},
    ]
    test_meta_reg = [{"frame_targets": [0.5, 1.5, 2.5, 3.5], "audio_length": 80}]

    res_reg = task_reg.evaluate(train_reps, train_meta_reg, test_reps, test_meta_reg)
    assert "mse" in res_reg
    assert "r2" in res_reg


def test_dictionary_target_missing_audio_length_raises():
    task_clf = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
    task_reg = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    reps = torch.randn(1, 4, 4)
    meta_clf = [{"frame_labels": {10: 0, 20: 1}}]
    meta_reg = [{"frame_targets": {10: 1.0, 20: 2.0}}]

    with pytest.raises(InvalidDatasetError, match="require 'audio_length'"):
        task_clf.evaluate(reps, meta_clf, reps, meta_clf)

    with pytest.raises(InvalidDatasetError, match="require 'audio_length'"):
        task_reg.evaluate(reps, meta_reg, reps, meta_reg)


def test_end_to_end_benchmark_frame_classification_arbitrary_and_dict(
    tmp_path,
):
    class CustomArbitraryFrameClfDataset(Dataset):
        def __init__(self, num_samples: int, is_dict: bool):
            self.num_samples = num_samples
            self.is_dict = is_dict

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> dict[str, Any]:
            if self.is_dict:
                # Dictionary targets with integer sample indices
                labels = {50: 0, 150: 1, 250: 0}
            else:
                # Arbitrary-length 7-element sequence (audio_length is 320)
                labels = [0, 1, 0, 1, 0, 1, 0]
            return {
                "waveform": torch.randn(1, 320),
                "sample_rate": 16000,
                "metadata": {"frame_labels": labels},
            }

    model = DummyAudioModel(representation_dim=16)
    task = FrameClassificationTask(classifier=KNNBenchmarkClassifier())

    # 1. Arbitrary sequence targets
    bench_seq = AudioBenchmark(
        dataset=CustomArbitraryFrameClfDataset(num_samples=6, is_dict=False),
        testset=CustomArbitraryFrameClfDataset(num_samples=4, is_dict=False),
        task=task,
        dataset_name="custom_seq_clf",
        cache_dir=tmp_path / "clf_seq",
        enable_cache=True,
    )
    res_seq = bench_seq.run(
        model=model,
        model_name="dummy_clf_seq",
        batch_size=2,
        device="cpu",
    )
    assert res_seq.task_name == "frame_classification"
    assert "accuracy" in res_seq.metrics
    assert 0.0 <= res_seq.metrics["accuracy"] <= 1.0

    # 2. Dictionary-mapped targets
    bench_dict = AudioBenchmark(
        dataset=CustomArbitraryFrameClfDataset(num_samples=6, is_dict=True),
        testset=CustomArbitraryFrameClfDataset(num_samples=4, is_dict=True),
        task=task,
        dataset_name="custom_dict_clf",
        cache_dir=tmp_path / "clf_dict",
        enable_cache=True,
    )
    res_dict = bench_dict.run(
        model=model,
        model_name="dummy_clf_dict",
        batch_size=2,
        device="cpu",
    )
    assert res_dict.task_name == "frame_classification"
    assert "accuracy" in res_dict.metrics
    assert 0.0 <= res_dict.metrics["accuracy"] <= 1.0


def test_end_to_end_benchmark_frame_regression_arbitrary_and_dict(tmp_path):
    class CustomArbitraryFrameRegDataset(Dataset):
        def __init__(self, num_samples: int, is_dict: bool):
            self.num_samples = num_samples
            self.is_dict = is_dict

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> dict[str, Any]:
            if self.is_dict:
                targets = {40: 1.2, 160: 2.8, 280: 4.1}
            else:
                targets = [1.0, 1.5, 2.0, 2.5, 3.0]
            return {
                "waveform": torch.randn(1, 320),
                "sample_rate": 16000,
                "metadata": {"frame_targets": targets},
            }

    model = DummyAudioModel(representation_dim=16)
    task = FrameRegressionTask(regressor=LinearBenchmarkRegressor())

    # 1. Arbitrary sequence targets
    bench_seq = AudioBenchmark(
        dataset=CustomArbitraryFrameRegDataset(num_samples=6, is_dict=False),
        testset=CustomArbitraryFrameRegDataset(num_samples=4, is_dict=False),
        task=task,
        dataset_name="custom_seq_reg",
        cache_dir=tmp_path / "reg_seq",
        enable_cache=True,
    )
    res_seq = bench_seq.run(
        model=model,
        model_name="dummy_reg_seq",
        batch_size=2,
        device="cpu",
    )
    assert res_seq.task_name == "frame_regression"
    assert "mse" in res_seq.metrics
    assert "r2" in res_seq.metrics

    # 2. Dictionary-mapped targets
    bench_dict = AudioBenchmark(
        dataset=CustomArbitraryFrameRegDataset(num_samples=6, is_dict=True),
        testset=CustomArbitraryFrameRegDataset(num_samples=4, is_dict=True),
        task=task,
        dataset_name="custom_dict_reg",
        cache_dir=tmp_path / "reg_dict",
        enable_cache=True,
    )
    res_dict = bench_dict.run(
        model=model,
        model_name="dummy_reg_dict",
        batch_size=2,
        device="cpu",
    )
    assert res_dict.task_name == "frame_regression"
    assert "mse" in res_dict.metrics
    assert "r2" in res_dict.metrics
