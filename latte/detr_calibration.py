"""DETR calibration semantics from the author's original script."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .detr_utils import create_detr_latte_attention


def cosine_loss(prediction, target):
    prediction = prediction / prediction.norm(dim=1, keepdim=True)
    target = target / target.norm(dim=1, keepdim=True)
    return 1 - (prediction * target).sum(dim=1).mean()


def calibrate_layer(original_attention, model_config, inputs, targets, settings):
    module = create_detr_latte_attention(original_attention, model_config)
    module.to(settings["device"])
    high, low = [], []
    for name, parameter in module.named_parameters():
        (low if "o_proj" in name or "v_proj" in name else high).append(parameter)
    optimizer = torch.optim.Adam([
        {"params": high, "lr": float(settings["learning_rate"])},
        {"params": low, "lr": float(settings["learning_rate"])},
    ])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    cosine_total, steps = 0.0, 0
    for _ in tqdm(range(settings["epochs"]), desc="DETR calibration"):
        for x, y in loader:
            optimizer.zero_grad()
            prediction, _ = module(x.float().to(settings["device"]))
            target = y.float().to(settings["device"])
            loss = cosine_loss(prediction, target)
            cosine_total += loss.item()
            loss = loss + nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            steps += 1
    mse_total, batches = 0.0, 0
    # Preserve the original shuffled final-MSE pass and training-average cosine.
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    with torch.no_grad():
        for x, y in loader:
            prediction, _ = module(x.float().to(settings["device"]))
            mse_total += nn.functional.mse_loss(
                prediction, y.float().to(settings["device"])
            ).item()
            batches += 1
    return module.cpu(), cosine_total / steps, mse_total / batches
