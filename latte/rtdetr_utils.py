"""Utilities for creating and installing LATTE layers in RT-DETR."""

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import torch

from .attention import RTDetrLATTEAttention


def get_decoder_layers(model):
    return model.model.decoder.layers


def create_rtdetr_latte_attention(original_attention, config):
    module = RTDetrLATTEAttention(
        hidden_size=config.d_model,
        num_attention_heads=config.decoder_attention_heads,
        dropout=0.0,
        bias=original_attention.q_proj.bias is not None,
    )
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(module, projection).load_state_dict(
            getattr(original_attention, projection).state_dict()
        )
    return module


def replace_rtdetr_layers(
    model,
    checkpoint: Union[str, Path, Mapping],
    layers: Optional[Iterable[int]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    records = checkpoint["layers"]
    selected = sorted(int(index) for index in (layers if layers is not None else records))
    decoder_layers = get_decoder_layers(model)
    for index in selected:
        record = records[index] if index in records else records[str(index)]
        module = RTDetrLATTEAttention(
            hidden_size=model.config.d_model,
            num_attention_heads=model.config.decoder_attention_heads,
            dropout=0.0,
            bias=decoder_layers[index].self_attn.q_proj.bias is not None,
        )
        module.load_state_dict(record["state_dict"])
        if device is not None:
            module.to(device)
        decoder_layers[index].self_attn = module
    return model


def select_layers_by_score(
    checkpoint: Mapping,
    num_layers: int,
    cosine_weight: float = 1.0,
    mse_weight: float = 10.0,
):
    """Select layers with the lowest weighted calibration errors."""
    records = checkpoint["layers"]
    if not 0 <= num_layers <= len(records):
        raise ValueError(f"num_layers must be between 0 and {len(records)}")
    ranked = sorted(
        (
            int(index),
            cosine_weight * record["cosine_loss"] + mse_weight * record["mse_loss"],
        )
        for index, record in records.items()
    )
    ranked.sort(key=lambda item: item[1])
    return sorted(index for index, _ in ranked[:num_layers])
