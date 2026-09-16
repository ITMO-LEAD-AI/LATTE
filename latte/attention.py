"""LATTE attention for Hugging Face RT-DETR decoder layers."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class RTDetrLATTEAttention(nn.Module):
    """Linearized RT-DETR self-attention using TTE and JPN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.is_causal = False
        self.eps = 1e-6

        self.alpha = nn.Parameter(torch.ones(1))
        self.delta_q = nn.Parameter(torch.ones(1))
        self.delta_k = nn.Parameter(torch.ones(1))
        self.ones_scale = nn.Parameter(torch.tensor(1.0))

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _tte(self, states: torch.Tensor) -> torch.Tensor:
        return (states + self.alpha).square() * torch.exp(1 - self.alpha.square()) * 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        del attention_mask, kwargs
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        query_key_input = (
            hidden_states + position_embeddings
            if position_embeddings is not None
            else hidden_states
        )

        query = self.q_proj(query_key_input).view(head_shape).transpose(1, 2)
        key = self.k_proj(query_key_input).view(head_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        query = query / (query.norm(dim=-1, keepdim=True) + self.eps)
        key = key / (key.norm(dim=-1, keepdim=True) + self.eps)
        query = self._tte(query)
        key = self._tte(key)

        joint_norm = (
            query.norm(dim=-1, keepdim=True) + self.eps
        ) * (key.norm(dim=-1, keepdim=True) + self.eps)
        query = self.delta_q * query / joint_norm
        key = self.delta_k * key / joint_norm

        ones = torch.ones_like(query[..., :1]) * self.ones_scale
        query = torch.cat((query, ones), dim=-1)
        key = torch.cat((key, ones), dim=-1)
        value = F.pad(value, (0, 1), mode="constant", value=1)

        key_value = key.transpose(-1, -2) @ value
        numerator_and_denominator = query @ key_value
        attention_output = numerator_and_denominator[..., :-1] / (
            numerator_and_denominator[..., -1:] + self.eps
        )
        attention_output = attention_output.transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(attention_output.contiguous()), None
