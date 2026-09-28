"""Dataset wrapper for generating frame-level voice activity detection (VAD) labels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from audio_benchmark.wrappers.base import BaseMetadataWrapper
from audio_benchmark.wrappers.utils import resample_waveform


class VoiceActivityWrapper(BaseMetadataWrapper):
    """Dynamically adds sample-by-sample voice activity labels to audio datasets.

    For each input sample in the raw audio waveform of length T, generates a binary
    indicator: 1 if someone is speaking, and 0 if no one is speaking.
    The resulting sequence has length T, perfectly matching `FrameClassificationTask`.

    Parameters
    ----------
    dataset : Dataset
        The underlying audio dataset returning standard 3-key dicts.
    method : str, default "silero"
        VAD method to use:
        - "silero": Deep learning VAD using Silero VAD (state-of-the-art accuracy).
        - "energy": Lightweight, zero-model short-time energy detector.
    key : str, default "frame_labels"
        Metadata key where the binary frame labels sequence will be stored.
    threshold : float, default 0.5
        Speech detection sensitivity threshold.
    cache : bool, default True
        If True, caches generated VAD labels in-memory.
    overwrite : bool, default False
        If False, reuses existing non-empty `metadata[key]`.
    model : Any, default None
        Pre-loaded Silero VAD model instance. If provided, bypasses loading.
    vad_fn : Callable[[torch.Tensor, int], list[int]] | None, default None
        Optional custom VAD callable returning a list of 0s and 1s of length T.
    return_tensor : bool, default False
        If True, stores VAD as a 1D `torch.Tensor`. If False (default), stores as
        a standard Python `list[int]`.
    """

    def __init__(
        self,
        dataset: Dataset,
        method: str = "silero",
        key: str = "frame_labels",
        threshold: float = 0.5,
        cache: bool = True,
        overwrite: bool = False,
        model: Any = None,
        vad_fn: Callable[[torch.Tensor, int], list[int]] | None = None,
        return_tensor: bool = False,
    ) -> None:
        super().__init__(dataset=dataset, key=key, cache=cache, overwrite=overwrite)
        if method not in ("silero", "energy"):
            raise ValueError(
                f"Unknown VAD method: '{method}'. Must be 'silero' or 'energy'."
            )
        self.method = method
        self.threshold = threshold
        self._model = model
        self.vad_fn = vad_fn
        self.return_tensor = return_tensor

    def _get_silero_model(self) -> Any:
        if self._model is None:
            try:
                import silero_vad

                self._model = silero_vad.load_silero_vad()
            except ImportError:
                # Fallback to torch.hub if silero-vad package is not directly installed
                self._model = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    onnx=False,
                    trust_repo=True,
                )
        return self._model

    def _generate_silero(
        self, waveform: torch.Tensor, sample_rate: int, orig_len: int
    ) -> torch.Tensor:
        """Run Silero VAD and project timestamps back to sample-level binary mask of length orig_len."""
        try:
            import silero_vad

            get_speech_timestamps = silero_vad.get_speech_timestamps
        except ImportError:
            from silero_vad.utils_vad import get_speech_timestamps  # type: ignore

        model = self._get_silero_model()

        # Resample to 16 kHz if needed
        audio_16k = resample_waveform(waveform, orig_sr=sample_rate, target_sr=16000)
        audio_1d = audio_16k.squeeze().float().detach().cpu()
        len_16k = audio_1d.shape[0]

        speech_intervals = get_speech_timestamps(
            audio_1d,
            model,
            threshold=self.threshold,
            sampling_rate=16000,
            return_seconds=False,
        )

        mask_16k = torch.zeros(len_16k, dtype=torch.float32)
        for interval in speech_intervals:
            start = max(0, min(interval["start"], len_16k))
            end = max(start, min(interval["end"], len_16k))
            mask_16k[start:end] = 1.0

        if len_16k == orig_len:
            return mask_16k.long()

        # Interpolate binary mask back to exact original audio length T
        mask_orig = (
            F.interpolate(
                mask_16k.view(1, 1, -1),
                size=orig_len,
                mode="nearest",
            )
            .view(-1)
            .long()
        )
        return mask_orig

    def _generate_energy(self, waveform: torch.Tensor, orig_len: int) -> torch.Tensor:
        """Lightweight short-time energy voice activity detector."""
        audio_1d = waveform.squeeze().float().detach().cpu()
        frame_length = min(512, orig_len)
        hop_length = max(1, frame_length // 2)

        if orig_len < frame_length or frame_length <= 0:
            avg_energy = float(torch.mean(audio_1d**2))
            is_speech = 1 if avg_energy > self.threshold else 0
            return torch.full((orig_len,), is_speech, dtype=torch.long)

        frames = audio_1d.unfold(0, frame_length, hop_length)
        frame_energies = torch.mean(frames**2, dim=-1)

        # Dynamic / absolute thresholding
        frame_vad = (frame_energies > self.threshold).float()

        mask_orig = (
            F.interpolate(
                frame_vad.view(1, 1, -1),
                size=orig_len,
                mode="nearest",
            )
            .view(-1)
            .long()
        )
        return mask_orig

    def generate(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        metadata: dict[str, Any],
    ) -> list[int] | torch.Tensor:
        """Generate binary voice activity labels for each sample in the waveform."""
        orig_len = waveform.squeeze().shape[0]

        if self.vad_fn is not None:
            custom_res = self.vad_fn(waveform, sample_rate)
            if len(custom_res) != orig_len:
                raise ValueError(
                    f"Custom VAD function returned {len(custom_res)} frames, "
                    f"but audio length is {orig_len}."
                )
            return (
                torch.as_tensor(custom_res, dtype=torch.long)
                if self.return_tensor
                else list(custom_res)
            )

        if self.method == "silero":
            vad_mask = self._generate_silero(waveform, sample_rate, orig_len)
        else:
            vad_mask = self._generate_energy(waveform, orig_len)

        return vad_mask if self.return_tensor else vad_mask.tolist()
