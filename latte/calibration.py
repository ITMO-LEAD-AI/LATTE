"""Activation collection and layer-wise RT-DETR LATTE calibration."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .rtdetr_utils import create_rtdetr_latte_attention, get_decoder_layers


def cosine_distance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = nn.functional.normalize(prediction, dim=1)
    target = nn.functional.normalize(target, dim=1)
    return 1 - (prediction * target).sum(dim=1).mean()


class RTDetrActivationCollector(AbstractContextManager):
    """Collect inputs, positions, and outputs of decoder self-attention."""

    def __init__(self, model, layer_indices: Iterable[int]):
        self.inputs: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
        self.positions: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
        self.outputs: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
        self.handles = []
        for index in self.inputs:
            attention = get_decoder_layers(model)[index].self_attn
            self.handles.append(
                attention.register_forward_pre_hook(self._input_hook(index), with_kwargs=True)
            )
            self.handles.append(
                attention.o_proj.register_forward_hook(self._output_hook(index))
            )

    def _input_hook(self, index):
        def hook(_module, args, kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                raise RuntimeError("Could not capture RT-DETR self-attention input")
            position_embeddings = kwargs.get(
                "position_embeddings", args[2] if len(args) > 2 else None
            )
            if position_embeddings is None:
                position_embeddings = torch.zeros_like(hidden_states)
            if position_embeddings.dim() == 2:
                position_embeddings = position_embeddings.unsqueeze(0).expand_as(hidden_states)
            self.inputs[index].append(
                hidden_states.detach().to(device="cpu", dtype=torch.bfloat16)
            )
            self.positions[index].append(
                position_embeddings.detach().to(device="cpu", dtype=torch.bfloat16)
            )
        return hook

    def _output_hook(self, index):
        def hook(_module, _inputs, output):
            self.outputs[index].append(
                output.detach().to(device="cpu", dtype=torch.bfloat16)
            )
        return hook

    def tensors(self, index):
        return (
            torch.cat(self.inputs[index]),
            torch.cat(self.positions[index]),
            torch.cat(self.outputs[index]),
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@dataclass
class CalibrationResult:
    module: nn.Module
    cosine_loss: float
    mse_loss: float


def configure_trainable_parameters(module: nn.Module, mode: str) -> None:
    if mode == "all":
        for parameter in module.parameters():
            parameter.requires_grad = True
        return
    if mode == "latte_only":
        names = {"alpha", "delta_q", "delta_k", "ones_scale"}
        for name, parameter in module.named_parameters():
            parameter.requires_grad = name in names
        return
    raise ValueError("trainable_parameters must be 'all' or 'latte_only'")


def calibrate_rtdetr_layer(
    original_attention,
    config,
    teacher_inputs,
    position_embeddings,
    teacher_outputs,
    settings,
):
    device = torch.device(settings["device"])
    module = create_rtdetr_latte_attention(original_attention, config).to(device)
    configure_trainable_parameters(module, settings.get("trainable_parameters", "all"))
    dataset = TensorDataset(teacher_inputs, position_embeddings, teacher_outputs)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(
        (parameter for parameter in module.parameters() if parameter.requires_grad),
        lr=float(settings["learning_rate"]),
    )
    cosine_weight = float(settings.get("cosine_weight", 1.0))
    mse_weight = float(settings.get("mse_weight", 1.0))

    module.train()
    for _ in tqdm(range(settings["epochs"]), desc="Calibrating layer", leave=False):
        for inputs, positions, targets in loader:
            inputs = inputs.float().to(device)
            positions = positions.float().to(device)
            targets = targets.float().to(device)
            prediction, _ = module(inputs, position_embeddings=positions)
            loss = cosine_weight * cosine_distance_loss(prediction, targets)
            loss += mse_weight * nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    module.eval()
    cosine_total = mse_total = 0.0
    batches = 0
    evaluation_loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False)
    with torch.no_grad():
        for inputs, positions, targets in evaluation_loader:
            prediction, _ = module(
                inputs.float().to(device),
                position_embeddings=positions.float().to(device),
            )
            targets = targets.float().to(device)
            cosine_total += cosine_distance_loss(prediction, targets).item()
            mse_total += nn.functional.mse_loss(prediction, targets).item()
            batches += 1
    return CalibrationResult(
        module=module.cpu(),
        cosine_loss=cosine_total / batches,
        mse_loss=mse_total / batches,
    )
