"""DETR calibration, preserving the author's original preprocessing and losses."""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CocoDetection
from transformers import DetrForObjectDetection, RTDetrImageProcessor
from tqdm import tqdm
from .detr_calibration import calibrate_layer


class ProcessedCocoDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset, self.processor = dataset, processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return {k: v.squeeze(0) for k, v in
                self.processor(images=image, return_tensors="pt").items()}


def collate_inputs(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def run_detr_calibration(config):
    settings = config["calibration"]
    torch.manual_seed(int(settings.get("seed", 42)))
    np.random.seed(int(settings.get("seed", 42)))
    if settings.get("trainable_parameters", "all") != "all":
        raise ValueError("Original DETR calibration trains all parameters.")
    if settings.get("cosine_weight", 1.0) != 1.0 or settings.get("mse_weight", 1.0) != 1.0:
        raise ValueError("Original DETR calibration uses cosine + MSE with unit weights.")
    device = torch.device(settings["device"])
    model_name = config["model"]["name"]
    # Deliberate: this is the calibration processor used in the original script.
    processor = RTDetrImageProcessor.from_pretrained(
        config["model"].get("calibration_processor", "PekingU/rtdetr_r50vd"))
    model = DetrForObjectDetection.from_pretrained(model_name).to(device).eval()
    ds = config["dataset"]
    dataset = CocoDetection(root=ds["images"], annFile=ds["annotations"])
    count = min(int(ds["num_samples"]), len(dataset))
    if count <= 0 or int(settings["epochs"]) <= 0:
        raise ValueError("num_samples and epochs must be positive")
    loader = DataLoader(
        ProcessedCocoDataset(Subset(dataset, range(count)), processor),
        batch_size=settings["collection_batch_size"], shuffle=False,
        collate_fn=collate_inputs)
    layers = model.model.decoder.layers
    selected = config["linearization"]["layers"]
    indices = list(range(len(layers))) if selected == "all" else list(selected)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(layers) for i in indices):
        raise ValueError("Invalid or duplicate layer indices")
    chunk_size = int(settings["chunk_size"])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    records = {}
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        captured = {}
        inputs, targets = {i: [] for i in chunk}, {i: [] for i in chunk}
        handles = []
        def make_hook(index):
            def hook(module, args, output):
                captured[index] = output.detach()
            return hook
        try:
            for i in chunk:
                handles.append(layers[i].self_attn.o_proj.register_forward_hook(make_hook(i)))
            for batch in tqdm(loader, desc=f"DETR teacher layers {chunk}"):
                with torch.no_grad():
                    output = model(**{k: v.to(device) for k, v in batch.items()},
                                   output_hidden_states=True)
                for i in chunk:
                    inputs[i].append(output.decoder_hidden_states[i].detach().to(
                        device="cpu", dtype=torch.bfloat16))
                    targets[i].append(captured[i].to(device="cpu", dtype=torch.bfloat16))
        finally:
            for handle in handles:
                handle.remove()
        captured.clear()
        del output
        for i in chunk:
            module, cos, mse = calibrate_layer(
                layers[i].self_attn, model.config,
                torch.cat(inputs[i]), torch.cat(targets[i]), settings)
            records[i] = {"state_dict": module.state_dict(),
                          "cosine_loss": cos, "mse_loss": mse}
            print(f"Layer {i}: training-average cosine={cos:.6f}, final MSE={mse:.6f}")
        del inputs, targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checkpoint = {"format_version": 2, "method": "LATTE",
                  "model_name": model_name, "model_type": "detr_decoder",
                  "implementation": "detr_original_v1", "config": config,
                  "layers": records}
    path = Path(config["output"]["checkpoint_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved {path}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Calibrate DETR using the original LATTE recipe")
    parser.add_argument("--config", required=True)
    run_detr_calibration(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
