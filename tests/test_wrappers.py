"""Unit and integration tests for dynamic dataset wrappers."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    BaseMetadataWrapper,
    CTCSequenceProbe,
    DummyAudioDataset,
    DummyAudioModel,
    FrameClassificationTask,
    KNNBenchmarkClassifier,
    SequenceModelingTask,
    validate_dataset,
)


class SimpleAudioDataset(Dataset):
    """Minimal dataset for wrapper testing."""

    def __init__(self, samples: list[tuple[torch.Tensor, int, dict]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav, sr, meta = self.samples[idx]
        return {
            "waveform": wav,
            "sample_rate": sr,
            "metadata": dict(meta),
        }


def test_base_wrapper_caching_and_overwrite():
    class CountingWrapper(BaseMetadataWrapper):
        def __init__(
            self, dataset: Dataset, cache: bool = True, overwrite: bool = False
        ):
            super().__init__(
                dataset=dataset, key="count", cache=cache, overwrite=overwrite
            )
            self.call_count = 0

        def generate(self, waveform, sample_rate, metadata):
            self.call_count += 1
            return self.call_count

    data = [
        (torch.zeros(16000), 16000, {"existing": 1}),
        (torch.zeros(16000), 16000, {"existing": 2}),
    ]
    raw_ds = SimpleAudioDataset(data)

    # 1. Test caching enabled
    wrapper = CountingWrapper(raw_ds, cache=True)
    assert len(wrapper) == 2

    item0_first = wrapper[0]
    assert item0_first["metadata"]["count"] == 1
    assert wrapper.call_count == 1

    item0_second = wrapper[0]
    assert item0_second["metadata"]["count"] == 1
    assert wrapper.call_count == 1  # Reused from cache!

    wrapper.clear_cache()
    item0_third = wrapper[0]
    assert item0_third["metadata"]["count"] == 2
    assert wrapper.call_count == 2

    # 2. Test precompute
    wrapper.precompute()
    assert wrapper.call_count == 3  # computed item 1 as well
    assert wrapper[1]["metadata"]["count"] == 3

    # 3. Test overwrite behavior
    data_with_key = [(torch.zeros(16000), 16000, {"count": 99})]
    ds_with_key = SimpleAudioDataset(data_with_key)

    wrapper_no_overwrite = CountingWrapper(ds_with_key, overwrite=False)
    assert wrapper_no_overwrite[0]["metadata"]["count"] == 99
    assert wrapper_no_overwrite.call_count == 0

    wrapper_overwrite = CountingWrapper(ds_with_key, overwrite=True)
    assert wrapper_overwrite[0]["metadata"]["count"] == 1
    assert wrapper_overwrite.call_count == 1


def test_wrapper_chaining_and_task_execution():
    class DummyFrameLabelWrapper(BaseMetadataWrapper):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset=dataset, key="frame_labels")

        def generate(self, waveform, sample_rate, metadata):
            length = waveform.squeeze().shape[0] if waveform.ndim > 1 else len(waveform)
            return [0] * length

    class DummyTokenWrapper(BaseMetadataWrapper):
        def __init__(self, dataset: Dataset):
            super().__init__(dataset=dataset, key="tokens")

        def generate(self, waveform, sample_rate, metadata):
            return ["a", "b", "c"]

    raw_dataset = DummyAudioDataset(
        num_samples=10, audio_length=16000, name="wrapped_dummy_train"
    )
    raw_testset = DummyAudioDataset(
        num_samples=4, audio_length=16000, name="wrapped_dummy_test"
    )

    # Chain custom wrappers together
    ds_wrapped = DummyTokenWrapper(DummyFrameLabelWrapper(raw_dataset))
    test_wrapped = DummyTokenWrapper(DummyFrameLabelWrapper(raw_testset))

    validate_dataset(ds_wrapped)
    validate_dataset(test_wrapped)

    sample = ds_wrapped[0]
    assert "tokens" in sample["metadata"]
    assert "frame_labels" in sample["metadata"]
    assert len(sample["metadata"]["frame_labels"]) == 16000

    model = DummyAudioModel(representation_dim=32)

    # Test FrameClassificationTask on frame_labels
    vad_task = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), label_key="frame_labels"
    )
    benchmark_vad = AudioBenchmark(
        dataset=ds_wrapped,
        testset=test_wrapped,
        task=vad_task,
        cache_dir=".test_wrapper_cache",
        enable_cache=False,
    )
    vad_res = benchmark_vad.run(
        model=model,
        model_name="test_model",
        batch_size=4,
    )
    assert "accuracy" in vad_res.metrics
    assert vad_res.metrics["accuracy"] >= 0.0

    # Test SequenceModelingTask on tokens
    seq_task = SequenceModelingTask(
        probe=CTCSequenceProbe(epochs=5, batch_size=4), target_key="tokens"
    )
    benchmark_seq = AudioBenchmark(
        dataset=ds_wrapped,
        testset=test_wrapped,
        task=seq_task,
        cache_dir=".test_wrapper_cache",
        enable_cache=False,
    )
    seq_res = benchmark_seq.run(
        model=model,
        model_name="test_model",
    )
    assert "token_error_rate" in seq_res.metrics
    assert seq_res.metrics["token_error_rate"] >= 0.0
