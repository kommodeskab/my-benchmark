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
    PhonemeWrapper,
    SequenceModelingTask,
    VoiceActivityWrapper,
    WhisperTranscriptionWrapper,
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


def test_whisper_wrapper_with_transcribe_fn():
    data = [
        (torch.zeros(16000), 16000, {}),
        (torch.zeros(24000), 24000, {}),
    ]
    raw_ds = SimpleAudioDataset(data)

    def mock_transcribe(wav, sr):
        assert wav.ndim in (1, 2)
        assert sr in (16000, 24000)
        return "the quick brown fox"

    wrapper = WhisperTranscriptionWrapper(
        raw_ds,
        key="transcription",
        transcribe_fn=mock_transcribe,
    )
    validate_dataset(wrapper)

    item0 = wrapper[0]
    assert item0["metadata"]["transcription"] == "the quick brown fox"
    assert "waveform" in item0
    assert "sample_rate" in item0


def test_whisper_wrapper_real_small_inference():
    # Test real Whisper small model inference on silent audio
    wav = torch.zeros(16000, dtype=torch.float32)
    raw_ds = SimpleAudioDataset([(wav, 16000, {})])

    wrapper = WhisperTranscriptionWrapper(
        raw_ds,
        model_name="small",
        device="cpu",
        language="en",
    )
    item = wrapper[0]
    assert "transcription" in item["metadata"]
    assert isinstance(item["metadata"]["transcription"], str)


def test_voice_activity_wrapper_lengths_and_methods():
    T_16k = 16000
    T_24k = 24000

    # 1. Silero method on 16kHz
    wav_16k = torch.zeros(T_16k, dtype=torch.float32)
    raw_ds_16k = SimpleAudioDataset([(wav_16k, 16000, {})])
    vad_silero = VoiceActivityWrapper(raw_ds_16k, method="silero")

    item_silero = vad_silero[0]
    vad_labels = item_silero["metadata"]["frame_labels"]
    assert isinstance(vad_labels, list)
    assert len(vad_labels) == T_16k
    # All zeros for silent audio
    assert all(x in (0, 1) for x in vad_labels)
    assert sum(vad_labels) == 0

    # 2. Silero method on different sample rate (resampling to 16k and projecting back to T)
    wav_24k = torch.zeros(T_24k, dtype=torch.float32)
    raw_ds_24k = SimpleAudioDataset([(wav_24k, 24000, {})])
    vad_resampled = VoiceActivityWrapper(raw_ds_24k, method="silero")
    item_resampled = vad_resampled[0]
    assert len(item_resampled["metadata"]["frame_labels"]) == T_24k

    # 3. Energy method
    wav_signal = torch.zeros(8000, dtype=torch.float32)
    wav_signal[2000:6000] = 1.0  # signal burst in middle
    raw_ds_signal = SimpleAudioDataset([(wav_signal, 16000, {})])
    vad_energy = VoiceActivityWrapper(
        raw_ds_signal,
        method="energy",
        threshold=0.5,
        return_tensor=True,
    )
    item_energy = vad_energy[0]
    vad_tensor = item_energy["metadata"]["frame_labels"]
    assert isinstance(vad_tensor, torch.Tensor)
    assert vad_tensor.shape == (8000,)
    assert 1 in vad_tensor.tolist()
    assert 0 in vad_tensor.tolist()

    # 4. Custom vad_fn
    def custom_vad(wav, sr):
        length = wav.squeeze().shape[0]
        return [1] * length

    vad_custom = VoiceActivityWrapper(raw_ds_16k, vad_fn=custom_vad)
    assert sum(vad_custom[0]["metadata"]["frame_labels"]) == T_16k


def test_phoneme_wrapper_from_existing_text():
    data = [
        (torch.zeros(16000), 16000, {"transcription": "Hello, world!"}),
        (torch.zeros(16000), 16000, {"text": "A quick test."}),
    ]
    raw_ds = SimpleAudioDataset(data)

    wrapper = PhonemeWrapper(raw_ds, drop_punctuation=True, strip_stress=False)
    validate_dataset(wrapper)

    item0 = wrapper[0]
    tokens0 = item0["metadata"]["tokens"]
    assert isinstance(tokens0, list)
    assert len(tokens0) > 0
    # Ensure no punctuation characters in tokens
    assert all("," not in t and "!" not in t and " " not in t for t in tokens0)

    # Test strip_stress
    wrapper_no_stress = PhonemeWrapper(raw_ds, strip_stress=True)
    tokens_no_stress = wrapper_no_stress[0]["metadata"]["tokens"]
    assert all(not any(char.isdigit() for char in t) for t in tokens_no_stress)

    # Item 1 with "text" key
    item1 = wrapper[1]
    assert len(item1["metadata"]["tokens"]) > 0


def test_phoneme_wrapper_auto_transcribe_fallback():
    # When text is absent, PhonemeWrapper transcribes audio dynamically via Whisper
    data = [(torch.zeros(16000), 16000, {})]
    raw_ds = SimpleAudioDataset(data)

    def mock_transcribe(wav, sr):
        return "Speech recognized"

    wrapper = PhonemeWrapper(raw_ds, transcribe_fn=mock_transcribe)
    item = wrapper[0]
    # Check that transcription was recorded
    assert item["metadata"]["transcription"] == "Speech recognized"
    tokens = item["metadata"]["tokens"]
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_wrapper_chaining_and_task_execution():
    # 1. Create dummy datasets
    raw_dataset = DummyAudioDataset(
        num_samples=10, audio_length=16000, name="wrapped_dummy_train"
    )
    raw_testset = DummyAudioDataset(
        num_samples=4, audio_length=16000, name="wrapped_dummy_test"
    )

    # 2. Chain wrappers together
    def mock_transcribe(wav, sr):
        return "hello deep learning benchmark"

    ds_with_text = WhisperTranscriptionWrapper(
        raw_dataset, transcribe_fn=mock_transcribe
    )
    ds_with_phonemes = PhonemeWrapper(ds_with_text, drop_punctuation=True)
    ds_with_vad = VoiceActivityWrapper(ds_with_phonemes, method="energy", threshold=0.1)

    test_with_text = WhisperTranscriptionWrapper(
        raw_testset, transcribe_fn=mock_transcribe
    )
    test_with_phonemes = PhonemeWrapper(test_with_text, drop_punctuation=True)
    test_with_vad = VoiceActivityWrapper(
        test_with_phonemes, method="energy", threshold=0.1
    )

    validate_dataset(ds_with_vad)
    validate_dataset(test_with_vad)

    sample = ds_with_vad[0]
    assert "transcription" in sample["metadata"]
    assert "tokens" in sample["metadata"]
    assert "frame_labels" in sample["metadata"]
    assert len(sample["metadata"]["frame_labels"]) == 16000

    # 3. Test FrameClassificationTask directly on the VAD-annotated dataset
    model = DummyAudioModel(representation_dim=32)

    vad_task = FrameClassificationTask(
        classifier=KNNBenchmarkClassifier(), label_key="frame_labels"
    )
    benchmark_vad = AudioBenchmark(
        dataset=ds_with_vad,
        testset=test_with_vad,
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

    # 4. Test SequenceModelingTask directly on the phoneme-annotated dataset
    seq_task = SequenceModelingTask(
        probe=CTCSequenceProbe(epochs=5, batch_size=4), target_key="tokens"
    )
    benchmark_seq = AudioBenchmark(
        dataset=ds_with_vad,
        testset=test_with_vad,
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
