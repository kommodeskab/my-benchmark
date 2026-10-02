"""CTC and Attention-based Transformer Decoder models for variable-length sequence modeling."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Embedding / feature dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass adding positional encoding to input tensor.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Position-encoded tensor of shape (B, T, d_model).
        """
        # x is (B, T, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ==============================================================================
# 1. CTC Linear Projection Probe
# ==============================================================================


class CTCSequenceProbe(nn.Module):
    """Linear projection trained with Connectionist Temporal Classification (CTC) loss.

    The blank token is placed at index 0. Real target IDs [0..vocab_size-1]
    are shifted to [1..vocab_size].

    Args:
        in_dim: Feature dimension of input audio representations.
        vocab_size: Number of distinct target tokens (excluding CTC blank).
    """

    def __init__(self, in_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = 0
        self.num_classes = vocab_size + 1
        self.linear = nn.Linear(in_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Memory representations of shape (B, T', in_dim).

        Returns:
            torch.Tensor: Log probabilities of shape (T', B, num_classes) for CTCLoss.
        """
        logits = self.linear(x)  # (B, T', num_classes)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        # Transpose to (T', B, num_classes) as expected by PyTorch CTCLoss
        return log_probs.transpose(0, 1)

    def greedy_decode(self, x: torch.Tensor) -> list[list[int]]:
        """Greedy best-path decoding from representations x of shape (B, T', in_dim).

        Collapses consecutive repeats and drops blank tokens (index 0).

        Args:
            x: Representation sequence of shape (B, T', in_dim).

        Returns:
            list[list[int]]: Decoded token ID sequences per sample.
        """
        self.eval()
        with torch.inference_mode():
            logits = self.linear(x)  # (B, T', num_classes)
            preds = logits.argmax(dim=-1).cpu().numpy()  # (B, T')

        decoded_batch: list[list[int]] = []
        for seq in preds:
            collapsed: list[int] = []
            prev_token = None
            for tok in seq:
                if tok != prev_token:
                    if tok != self.blank_id:
                        # Shift index back: (tok - 1)
                        collapsed.append(int(tok - 1))
                    prev_token = tok
            decoded_batch.append(collapsed)

        return decoded_batch


# ==============================================================================
# 2. Attention-Based Transformer Decoder Probe
# ==============================================================================


class TransformerDecoderProbe(nn.Module):
    """Autoregressive Transformer Decoder with cross-attention over audio representations.

    Special tokens:
      - 0: <pad>
      - 1: <sos> (start of sequence)
      - 2: <eos> (end of sequence)
      - 3..vocab_size+2: user tokens

    Args:
        in_dim: Input feature dimension of audio representations.
        vocab_size: Size of target vocabulary.
        d_model: Decoder embedding and hidden dimension.
        nhead: Number of attention heads in multi-head attention.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Dimension of the feedforward network layer.
        dropout: Dropout probability.
    """

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

        # Memory projection from C' to d_model
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
    ) -> torch.Tensor:
        """Forward pass with teacher-forcing.

        Args:
            memory: Audio representations (B, T', C').
            tgt: Target token sequence with <sos> (B, S).
            tgt_mask: Causal mask of shape (S, S).
            tgt_key_padding_mask: Padding mask of shape (B, S).

        Returns:
            torch.Tensor: Logits of shape (B, S, total_tokens).
        """
        proj_memory = self.memory_proj(memory)  # (B, T', d_model)

        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        out = self.decoder(
            tgt=tgt_embed,
            memory=proj_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.output_proj(out)
        return logits

    def generate(self, memory: torch.Tensor, max_len: int = 50) -> list[list[int]]:
        """Greedy autoregressive decoding starting from <sos>.

        Args:
            memory: Audio representations tensor of shape (B, T', C').
            max_len: Maximum generation length in tokens.

        Returns:
            list[list[int]]: Generated token ID sequences.
        """
        self.eval()
        device = memory.device
        batch_size = memory.size(0)

        proj_memory = self.memory_proj(memory)

        # Start sequence: (B, 1) filled with SOS_ID
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

                out = self.decoder(tgt=tgt_embed, memory=proj_memory, tgt_mask=tgt_mask)
                next_logits = self.output_proj(out[:, -1, :])  # (B, total_tokens)
                next_tokens = next_logits.argmax(dim=-1)  # (B,)

                # Update finished status
                finished = finished | (next_tokens == self.EOS_ID)
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                if finished.all():
                    break

        # Decode tokens to list of original IDs
        decoded_batch: list[list[int]] = []
        gen_np = generated.cpu().numpy()

        for seq in gen_np:
            toks: list[int] = []
            for tok in seq[1:]:  # skip <sos>
                if tok == self.EOS_ID:
                    break
                if tok == self.PAD_ID:
                    continue
                # Shift back
                toks.append(int(tok - self.NUM_SPECIAL))
            decoded_batch.append(toks)

        return decoded_batch
