"""Dataset wrapper for generating text transcriptions using Whisper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_benchmark.wrappers.base import BaseMetadataWrapper
from audio_benchmark.wrappers.utils import resample_waveform


class WhisperTranscriptionWrapper(BaseMetadataWrapper):
    """Dynamically adds text transcriptions to audio samples using Whisper.

    By default, uses the OpenAI Whisper 'small' model. Audio waveforms of any
    sample rate are automatically resampled to 16 kHz as required by Whisper.

    Parameters
    ----------
    dataset : Dataset
        The underlying audio dataset returning standard 3-key dicts.
    model_name : str, default "small"
        Name/size of the Whisper model to load (e.g. "small", "base", "tiny").
    device : str | None, default None
        PyTorch device ('cuda', 'cpu', etc.). If None, automatically selects
        'cuda' if available, otherwise 'cpu'.
    language : str | None, default None
        Optional language code (e.g. 'en') to pass to Whisper. If None, Whisper
        auto-detects the spoken language.
    key : str, default "transcription"
        Key in `metadata` where the transcription string will be stored.
    cache : bool, default True
        If True, caches generated transcriptions in-memory so each sample is only
        transcribed once.
    overwrite : bool, default False
        If False, reuses existing non-empty `metadata[key]`.
    whisper_model : Any, default None
        Pre-loaded Whisper model instance. If provided, bypasses loading via `model_name`.
    transcribe_fn : Callable[[torch.Tensor, int], str] | None, default None
        Optional custom transcription callable accepting `(waveform, sample_rate)`
        and returning text. Useful for testing or custom Whisper inference pipelines.
    """

    def __init__(
        self,
        dataset: Dataset,
        model_name: str = "small",
        device: str | None = None,
        language: str | None = None,
        key: str = "transcription",
        cache: bool = True,
        overwrite: bool = False,
        whisper_model: Any = None,
        transcribe_fn: Callable[[torch.Tensor, int], str] | None = None,
    ) -> None:
        super().__init__(dataset=dataset, key=key, cache=cache, overwrite=overwrite)
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.language = language
        self._whisper_model = whisper_model
        self.transcribe_fn = transcribe_fn

    def _get_model(self) -> Any:
        if self._whisper_model is None:
            try:
                import whisper

                self._whisper_model = whisper.load_model(
                    self.model_name, device=self.device
                )
            except ImportError as e:
                raise ImportError(
                    "openai-whisper is required for WhisperTranscriptionWrapper. "
                    "Install it via: pip install openai-whisper"
                ) from e
        return self._whisper_model

    def generate(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        metadata: dict[str, Any],
    ) -> str:
        """Transcribe the audio waveform to text."""
        if self.transcribe_fn is not None:
            return self.transcribe_fn(waveform, sample_rate)

        model = self._get_model()

        # Resample to 16 kHz if necessary
        audio_16k = resample_waveform(waveform, orig_sr=sample_rate, target_sr=16000)
        audio_1d = audio_16k.squeeze().float().detach().cpu()

        # Normalize if necessary (avoid clipping / out-of-range floats)
        max_val = torch.max(torch.abs(audio_1d))
        if max_val > 1.0:
            audio_1d = audio_1d / max_val

        audio_np = audio_1d.numpy().astype(np.float32)

        use_fp16 = str(self.device).startswith("cuda") and torch.cuda.is_available()
        result = model.transcribe(
            audio_np,
            language=self.language,
            fp16=use_fp16,
            verbose=False,
        )
        return str(result.get("text", "")).strip()
