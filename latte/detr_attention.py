"""DETR attention preserved from the original author script."""
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from typing_extensions import Unpack
from transformers.utils import TransformersKwargs

class DetrLinearAttention(nn.Module):
    """
    Multi-headed self-attention from 'Attention Is All You Need' paper.

    In DETR, position embeddings are added to both queries and keys (but not values) in self-attention.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float=0.0, bias: bool=True):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = dropout
        self.is_causal = False
        self.eps = 1e-06
        self.alpha = nn.Parameter(torch.ones(1))
        self.delta_q = nn.Parameter(torch.ones(1))
        self.delta_k = nn.Parameter(torch.ones(1))
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ones_scale1 = nn.Parameter(torch.tensor(1.0))

    def linear_attention_forward_qtvit(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], scaling: float, dropout: float=0.0, eps: float=1e-06, ones_scale1=None) -> Tuple[torch.Tensor, None]:
        query_states = query_states / (query_states.norm(dim=-1, keepdim=True) + eps)
        key_states = key_states / (key_states.norm(dim=-1, keepdim=True) + eps)
        query_states = (query_states + self.alpha) ** 2 * torch.exp(1 - self.alpha ** 2) * 0.5
        key_states = (key_states + self.alpha) ** 2 * torch.exp(1 - self.alpha ** 2) * 0.5
        q_norm = query_states.norm(dim=-1, keepdim=True) + self.eps
        k_norm = key_states.norm(dim=-1, keepdim=True) + self.eps
        query_states = self.delta_q * query_states / (q_norm * k_norm)
        key_states = self.delta_k * key_states / (q_norm * k_norm)
        B, H, T, D = query_states.shape
        ones = torch.ones(B, H, T, 1, device=query_states.device)
        ones1 = ones * ones_scale1
        query_states = torch.cat([query_states, ones1], dim=-1)
        key_states = torch.cat([key_states, ones1], dim=-1)
        trans_k = key_states.transpose(-1, -2)
        kv = torch.matmul(trans_k, F.pad(value_states, (0, 1), value=1))
        attn_output = torch.matmul(query_states, kv)
        attn_output = attn_output[..., :-1] / (attn_output[..., -1:] + eps)
        return (attn_output, None)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_embeddings: torch.Tensor | None=None, **kwargs: Unpack[TransformersKwargs]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Position embeddings are added to both queries and keys (but not values).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_key_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states
        query_states = self.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        attn_output, _ = self.linear_attention_forward_qtvit(query_states=query_states, key_states=key_states, value_states=value_states, attention_mask=attention_mask, scaling=self.scaling, dropout=0.0 if not self.training else self.attention_dropout, ones_scale1=self.ones_scale1)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, None)

DetrLATTEAttention = DetrLinearAttention
