"""CTC and Attention-based Transformer Decoder probes for variable-length sequence modeling."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audio_benchmark.utils import PositionalEncoding, resolve_device


def _pad_representation_batch(
    reps: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Zero-pad variable-length 2D representation tensors in a batch to the longest sequence.

    Args:
        reps: Sequence of 2D tensors of shape (T'_i, C').

    Returns:
        tuple[torch.Tensor, list[int], torch.Tensor]:
            - padded_reps: Tensor of shape (B, max_T', C').
            - lengths: List of unpadded sequence lengths T'_i.
            - memory_key_padding_mask: Boolean tensor of shape (B, max_T') where True indicates padded positions.
    """
    lengths = [int(r.shape[0]) for r in reps]
    max_len = max(lengths) if lengths else 0
    batch_size = len(reps)

    padded_reps = [
        torch.nn.functional.pad(r, (0, 0, 0, max_len - r.shape[0]))
        if r.shape[0] < max_len
        else r
        for r in reps
    ]
    stacked = torch.stack(padded_reps, dim=0)

    memory_key_padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length < max_len:
            memory_key_padding_mask[i, length:] = True

    return stacked, lengths, memory_key_padding_mask


class BaseSequenceProbe(ABC):
    """Abstract base class for downstream sequence probes."""

    @abstractmethod
    def fit(
        self,
        X: list[torch.Tensor],
        targets: list[list[int]],
        vocab_size: int,
    ) -> BaseSequenceProbe:
        """Fit the probe on representation features X and integer token targets.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').
            targets: List of integer target sequences.
            vocab_size: Number of unique vocabulary tokens.

        Returns:
            BaseSequenceProbe: Fitted probe instance.
        """

    @abstractmethod
    def predict(self, X: list[torch.Tensor]) -> list[list[int]]:
        """Predict token ID sequences from representation features X.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').

        Returns:
            list[list[int]]: Decoded token ID sequences per sample.
        """


class _TorchCTCProbe(nn.Module):
    """Internal PyTorch linear projection for CTC loss."""

    def __init__(self, in_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = 0
        self.num_classes = vocab_size + 1
        self.linear = nn.Linear(in_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)

    def greedy_decode(
        self, x: torch.Tensor, lengths: Sequence[int] | None = None
    ) -> list[list[int]]:
        self.eval()
        with torch.inference_mode():
            logits = self.linear(x)
            preds = logits.argmax(dim=-1).cpu().numpy()

        decoded_batch: list[list[int]] = []
        for i, seq in enumerate(preds):
            valid_seq = seq[: lengths[i]] if lengths is not None else seq
            collapsed: list[int] = []
            prev_token = None
            for tok in valid_seq:
                if tok != prev_token:
                    if tok != self.blank_id:
                        collapsed.append(int(tok - 1))
                    prev_token = tok
            decoded_batch.append(collapsed)
        return decoded_batch


class _PairDataset(Dataset):
    """Dataset pairing representations with target token sequences."""

    def __init__(
        self, representations: Sequence[torch.Tensor], targets: list[list[int]]
    ):
        self.representations = representations
        self.targets = targets

    def __len__(self) -> int:
        return len(self.representations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        return self.representations[idx], self.targets[idx]


def _collate_ctc_batch(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, list[int], list[list[int]]]:
    """Collate batch for CTC training with dynamic representation zero-padding."""
    reps, lengths, _ = _pad_representation_batch([x[0] for x in batch])
    targets = [x[1] for x in batch]
    return reps, lengths, targets


class CTCSequenceProbe(BaseSequenceProbe):
    """Linear projection trained with Connectionist Temporal Classification (CTC) loss.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for probe training.
        lr: Learning rate for probe optimizer.
        device: Device to train on ('auto', 'cpu', 'cuda').
        show_progress: Whether to display a tqdm progress bar during training.
    """

    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str | torch.device = "auto",
        show_progress: bool = True,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device_str = device
        self.device = resolve_device(device)
        self.show_progress = show_progress
        self.model: _TorchCTCProbe | None = None

    def fit(
        self,
        X: list[torch.Tensor],
        targets: list[list[int]],
        vocab_size: int,
    ) -> CTCSequenceProbe:
        """Fit the CTC probe on features X and token targets.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').
            targets: List of integer target sequences.
            vocab_size: Number of distinct vocabulary tokens.

        Returns:
            CTCSequenceProbe: Fitted probe instance.
        """
        in_dim = X[0].shape[-1]
        self.model = _TorchCTCProbe(in_dim=in_dim, vocab_size=vocab_size).to(
            self.device
        )

        dataset = _PairDataset(X, targets)
        loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, len(dataset))),
            shuffle=True,
            collate_fn=_collate_ctc_batch,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.model.train()
        pbar = tqdm(
            range(self.epochs),
            desc="Training CTC Probe",
            disable=not self.show_progress,
            leave=False,
        )
        for _ in pbar:
            total_loss = 0.0
            for batch_reps, rep_lengths, batch_targets in loader:
                batch_reps = batch_reps.to(self.device)

                shifted_targets = [[tok + 1 for tok in seq] for seq in batch_targets]
                flat_targets = torch.tensor(
                    [tok for seq in shifted_targets for tok in seq],
                    dtype=torch.long,
                )
                input_lengths = torch.tensor(rep_lengths, dtype=torch.long)
                target_lengths = torch.tensor(
                    [len(seq) for seq in shifted_targets], dtype=torch.long
                )

                optimizer.zero_grad()
                log_probs = self.model(batch_reps)
                loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{total_loss / max(1, len(loader)):.4f}")

        return self

    def predict(self, X: list[torch.Tensor]) -> list[list[int]]:
        """Predict token ID sequences from representations X.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').

        Returns:
            list[list[int]]: Decoded token sequences.
        """
        if self.model is None:
            raise RuntimeError("CTCSequenceProbe has not been fitted yet.")
        if not X:
            return []
        results: list[list[int]] = []
        step = max(1, self.batch_size)
        for start in range(0, len(X), step):
            batch_slice = X[start : start + step]
            padded_reps, lengths, _ = _pad_representation_batch(batch_slice)
            results.extend(
                self.model.greedy_decode(padded_reps.to(self.device), lengths=lengths)
            )
        return results


class _TorchTransformerDecoder(nn.Module):
    """Internal PyTorch autoregressive Transformer Decoder."""

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    NUM_SPECIAL = 3

    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_tokens = vocab_size + self.NUM_SPECIAL
        self.d_model = d_model

        if in_dim != d_model:
            self.memory_proj = nn.Linear(in_dim, d_model)
        else:
            self.memory_proj = nn.Identity()

        self.token_embedding = nn.Embedding(
            self.total_tokens, d_model, padding_idx=self.PAD_ID
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.total_tokens)

    def forward(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proj_memory = self.memory_proj(memory)
        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        out = self.decoder(
            tgt=tgt_embed,
            memory=proj_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def generate(
        self,
        memory: torch.Tensor,
        max_len: int = 50,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        self.eval()
        device = memory.device
        batch_size = memory.size(0)

        proj_memory = self.memory_proj(memory)
        generated = torch.full(
            (batch_size, 1), self.SOS_ID, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.inference_mode():
            for _ in range(max_len):
                seq_len = generated.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=device, dtype=torch.bool
                )
                tgt_embed = self.token_embedding(generated) * math.sqrt(self.d_model)
                tgt_embed = self.pos_encoder(tgt_embed)

                out = self.decoder(
                    tgt=tgt_embed,
                    memory=proj_memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                next_logits = self.output_proj(out[:, -1, :])
                next_tokens = next_logits.argmax(dim=-1)

                finished = finished | (next_tokens == self.EOS_ID)
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                if finished.all():
                    break

        decoded_batch: list[list[int]] = []
        gen_np = generated.cpu().numpy()

        for seq in gen_np:
            toks: list[int] = []
            for tok in seq[1:]:
                if tok == self.EOS_ID:
                    break
                if tok == self.PAD_ID:
                    continue
                toks.append(int(tok - self.NUM_SPECIAL))
            decoded_batch.append(toks)

        return decoded_batch


def _collate_transformer_batch(
    batch: list[tuple[torch.Tensor, list[int]]],
    sos_id: int,
    eos_id: int,
    pad_id: int,
    num_special: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch with padded representations, teacher-forcing inputs, targets, and padding masks."""
    reps, _, memory_padding_mask = _pad_representation_batch(
        [item[0] for item in batch]
    )
    targets = [item[1] for item in batch]

    max_len = max(len(t) for t in targets) if targets else 0
    batch_size = len(batch)

    dec_input = torch.full((batch_size, max_len + 1), pad_id, dtype=torch.long)
    dec_target = torch.full((batch_size, max_len + 1), pad_id, dtype=torch.long)
    padding_mask = torch.zeros((batch_size, max_len + 1), dtype=torch.bool)

    for i, seq in enumerate(targets):
        shifted_seq = [tok + num_special for tok in seq]
        seq_len = len(shifted_seq)

        dec_input[i, 0] = sos_id
        if seq_len > 0:
            dec_input[i, 1 : seq_len + 1] = torch.tensor(shifted_seq, dtype=torch.long)

        dec_target[i, :seq_len] = torch.tensor(shifted_seq, dtype=torch.long)
        dec_target[i, seq_len] = eos_id
        padding_mask[i, seq_len + 1 :] = True

    return reps, dec_input, dec_target, padding_mask, memory_padding_mask


class TransformerDecoderProbe(BaseSequenceProbe):
    """Autoregressive Transformer Decoder probe with cross-attention.

    Args:
        d_model: Decoder embedding and hidden dimension.
        nhead: Number of attention heads in multi-head attention.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Dimension of feedforward network layer.
        dropout: Dropout probability.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate for probe optimizer.
        device: Device to train on ('auto', 'cpu', 'cuda').
        show_progress: Whether to display a tqdm progress bar during training.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str | torch.device = "auto",
        show_progress: bool = True,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device_str = device
        self.device = resolve_device(device)
        self.show_progress = show_progress
        self.model: _TorchTransformerDecoder | None = None
        self.max_gen_len = 50

    def fit(
        self,
        X: list[torch.Tensor],
        targets: list[list[int]],
        vocab_size: int,
    ) -> TransformerDecoderProbe:
        """Fit the Transformer Decoder probe on features X and token targets.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').
            targets: List of integer target sequences.
            vocab_size: Number of distinct vocabulary tokens.

        Returns:
            TransformerDecoderProbe: Fitted probe instance.
        """
        in_dim = X[0].shape[-1]
        self.model = _TorchTransformerDecoder(
            in_dim=in_dim,
            vocab_size=vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        max_target_len = max(len(t) for t in targets) if targets else 0
        self.max_gen_len = max_target_len + 15

        dataset = _PairDataset(X, targets)
        loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, len(dataset))),
            shuffle=True,
            collate_fn=lambda b: _collate_transformer_batch(
                b,
                sos_id=_TorchTransformerDecoder.SOS_ID,
                eos_id=_TorchTransformerDecoder.EOS_ID,
                pad_id=_TorchTransformerDecoder.PAD_ID,
                num_special=_TorchTransformerDecoder.NUM_SPECIAL,
            ),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=_TorchTransformerDecoder.PAD_ID)

        self.model.train()
        pbar = tqdm(
            range(self.epochs),
            desc="Training Transformer Probe",
            disable=not self.show_progress,
            leave=False,
        )
        for _ in pbar:
            total_loss = 0.0
            for (
                batch_reps,
                dec_in,
                dec_tgt,
                padding_mask,
                memory_padding_mask,
            ) in loader:
                batch_reps = batch_reps.to(self.device)
                dec_in = dec_in.to(self.device)
                dec_tgt = dec_tgt.to(self.device)
                padding_mask = padding_mask.to(self.device)
                memory_padding_mask = memory_padding_mask.to(self.device)

                s_len = dec_in.size(1)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    s_len, device=self.device, dtype=torch.bool
                )

                optimizer.zero_grad()
                logits = self.model(
                    memory=batch_reps,
                    tgt=dec_in,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=memory_padding_mask,
                )
                loss = criterion(
                    logits.reshape(-1, self.model.total_tokens), dec_tgt.reshape(-1)
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{total_loss / max(1, len(loader)):.4f}")

        return self

    def predict(self, X: list[torch.Tensor]) -> list[list[int]]:
        """Predict token ID sequences from representations X.

        Args:
            X: List of 2D representation tensors of shape (T'_i, C').

        Returns:
            list[list[int]]: Decoded token sequences.
        """
        if self.model is None:
            raise RuntimeError("TransformerDecoderProbe has not been fitted yet.")
        if not X:
            return []
        results: list[list[int]] = []
        step = max(1, self.batch_size)
        for start in range(0, len(X), step):
            batch_slice = X[start : start + step]
            padded_reps, _, memory_padding_mask = _pad_representation_batch(batch_slice)
            results.extend(
                self.model.generate(
                    padded_reps.to(self.device),
                    max_len=self.max_gen_len,
                    memory_key_padding_mask=memory_padding_mask.to(self.device),
                )
            )
        return results
