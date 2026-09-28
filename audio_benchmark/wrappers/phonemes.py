"""Dataset wrapper for generating phoneme token sequences using G2P and Whisper."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from audio_benchmark.wrappers.base import BaseMetadataWrapper
from audio_benchmark.wrappers.whisper import WhisperTranscriptionWrapper


class PhonemeWrapper(BaseMetadataWrapper):
    """Dynamically converts audio or text transcriptions to phoneme token sequences.

    Uses Grapheme-to-Phoneme (g2p_en) to produce phoneme lists (e.g. ['HH', 'AH0', 'L', 'OW1']).
    If text transcription is not already present in sample metadata, it automatically
    transcribes the raw audio waveform using Whisper ('small' model by default).

    The output list of phonemes matches the sequence format required by `SequenceModelingTask`.

    Parameters
    ----------
    dataset : Dataset
        The underlying audio dataset returning standard 3-key dicts.
    key : str, default "tokens"
        Metadata key where the list of phonemes will be stored.
    text_key : str, default "transcription"
        Key in `metadata` to check for text transcriptions.
    drop_punctuation : bool, default True
        If True, filters out non-phonemic punctuation marks and whitespace.
    strip_stress : bool, default False
        If True, strips numerical stress markers from ARPAbet phonemes
        (e.g., 'AH0' -> 'AH', 'OW1' -> 'OW').
    cache : bool, default True
        If True, caches generated phoneme lists in-memory.
    overwrite : bool, default False
        If False, reuses existing non-empty `metadata[key]`.
    phonemizer : Callable[[str], list[str]] | None, default None
        Optional custom phonemization callable accepting a text string and
        returning a list of phoneme strings.
    whisper_model : Any, default None
        Optional pre-loaded Whisper model instance used for automatic transcription
        if text is absent.
    transcribe_fn : Callable[[torch.Tensor, int], str] | None, default None
        Optional custom transcription callable used if text is absent.
    whisper_kwargs : dict[str, Any] | None, default None
        Keyword arguments passed to internal `WhisperTranscriptionWrapper`
        (e.g. `{"model_name": "small", "device": "cuda", "language": "en"}`).
    """

    def __init__(
        self,
        dataset: Dataset,
        key: str = "tokens",
        text_key: str = "transcription",
        drop_punctuation: bool = True,
        strip_stress: bool = False,
        cache: bool = True,
        overwrite: bool = False,
        phonemizer: Callable[[str], list[str]] | None = None,
        whisper_model: Any = None,
        transcribe_fn: Callable[[torch.Tensor, int], str] | None = None,
        whisper_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dataset=dataset, key=key, cache=cache, overwrite=overwrite)
        self.text_key = text_key
        self.drop_punctuation = drop_punctuation
        self.strip_stress = strip_stress
        self.phonemizer = phonemizer
        self._whisper_model = whisper_model
        self.transcribe_fn = transcribe_fn
        self.whisper_kwargs = whisper_kwargs or {}
        self._g2p = None
        self._whisper_wrapper = None

    def _get_g2p(self) -> Any:
        if self._g2p is None:
            try:
                import nltk

                # Ensure required NLTK resources are available without manual user steps
                for res in [
                    "averaged_perceptron_tagger_eng",
                    "cmudict",
                    "punkt",
                    "punkt_tab",
                ]:
                    try:
                        res_path = (
                            f"taggers/{res}"
                            if "tagger" in res
                            else f"corpora/{res}"
                            if res == "cmudict"
                            else f"tokenizers/{res}"
                        )
                        nltk.data.find(res_path)
                    except (LookupError, OSError):
                        nltk.download(res, quiet=True)

                from g2p_en import G2p

                self._g2p = G2p()
            except ImportError as e:
                raise ImportError(
                    "g2p-en and nltk are required for PhonemeWrapper. "
                    "Install them via: pip install g2p-en nltk"
                ) from e
        return self._g2p

    def _get_whisper_transcriber(self) -> WhisperTranscriptionWrapper:
        if self._whisper_wrapper is None:
            kwargs = dict(self.whisper_kwargs)
            kwargs.setdefault("model_name", "small")
            kwargs.setdefault("key", self.text_key)
            kwargs.setdefault("whisper_model", self._whisper_model)
            kwargs.setdefault("transcribe_fn", self.transcribe_fn)
            self._whisper_wrapper = WhisperTranscriptionWrapper(
                dataset=self.dataset,
                **kwargs,
            )
        return self._whisper_wrapper

    def _g2p_to_phonemes(self, text: str) -> list[str]:
        if not text.strip():
            return []
        g2p = self._get_g2p()
        tokens = g2p(text)
        phonemes: list[str] = []
        for t in tokens:
            if self.drop_punctuation and not re.match(r"^[A-Za-z0-9]+$", t):
                continue
            if self.strip_stress:
                t = re.sub(r"\d+", "", t)
            if t:
                phonemes.append(t)
        return phonemes

    def generate(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        metadata: dict[str, Any],
    ) -> list[str]:
        """Derive phonemes from existing metadata text or via automatic Whisper transcription."""
        # 1. Retrieve text if already in metadata
        text = None
        if metadata.get(self.text_key):
            text = str(metadata[self.text_key])
        else:
            for alt_key in ["transcription", "text", "transcript"]:
                if metadata.get(alt_key):
                    text = str(metadata[alt_key])
                    break

        # 2. If no text exists, automatically transcribe audio via Whisper
        if text is None:
            transcriber = self._get_whisper_transcriber()
            text = transcriber.generate(waveform, sample_rate, metadata)
            # Save transcription to metadata for downstream use
            metadata[self.text_key] = text

        # 3. Convert text to phonemes
        if self.phonemizer is not None:
            return self.phonemizer(text)

        return self._g2p_to_phonemes(text)
