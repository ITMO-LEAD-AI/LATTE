"""Utilities for creating and installing LATTE layers in DETR."""

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import torch
import numpy as np

from .detr_attention import DetrLATTEAttention


def load_detr_checkpoint(checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "transforms" in checkpoint:
        return {
            "model_name": "facebook/detr-resnet-50",
            "implementation": "detr_original_v1",
            "layers": {
                i: {"state_dict": sd, "cosine_loss": checkpoint["cos_loss"][i],
                    "mse_loss": checkpoint["mse_loss"][i]}
                for i, sd in enumerate(checkpoint["transforms"])
            },
        }
    if checkpoint.get("implementation") != "detr_original_v1":
        raise ValueError(
            "This DETR checkpoint was made by the earlier refactor. Use the original "
            "DETR_LATTE_Repo.pth or recalibrate with this release; RT-DETR checkpoints "
            "are unchanged."
        )
    return checkpoint


def select_layers_by_score(checkpoint, num_layers, cosine_weight=1.0, mse_weight=10.0):
    checkpoint = load_detr_checkpoint(checkpoint)
    records = checkpoint["layers"]
    indices = sorted(int(i) for i in records)
    if not 0 <= num_layers <= len(indices):
        raise ValueError("Requested more layers than the checkpoint contains")
    scores = []
    for i in indices:
        r = records[i] if i in records else records[str(i)]
        scores.append(cosine_weight * r["cosine_loss"] + mse_weight * r["mse_loss"])
    return [indices[j] for j in np.argsort(np.asarray(scores))[:num_layers]]


def get_decoder_layers(model):
    return model.model.decoder.layers


def create_detr_latte_attention(original_attention, config):
    module = DetrLATTEAttention(
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


def replace_detr_layers(
    model,
    checkpoint: Union[str, Path, Mapping],
    layers: Optional[Iterable[int]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    checkpoint = load_detr_checkpoint(checkpoint)
    records = checkpoint["layers"]
    selected = sorted(int(index) for index in (layers if layers is not None else records))
    decoder_layers = get_decoder_layers(model)
    for index in selected:
        record = records[index] if index in records else records[str(index)]
        module = create_detr_latte_attention(decoder_layers[index].self_attn, model.config)
        module.load_state_dict(record["state_dict"])
        if device is not None:
            module.to(device)
        decoder_layers[index].self_attn = module
    return model
