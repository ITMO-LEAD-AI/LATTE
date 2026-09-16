"""Configuration-driven RT-DETR calibration pipeline."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from .calibration import RTDetrActivationCollector, calibrate_rtdetr_layer
from .rtdetr_utils import get_decoder_layers


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_layers(value, number_of_layers):
    if value == "all":
        return list(range(number_of_layers))
    return [int(index) for index in value]


class ProcessedCocoDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        inputs = self.processor(images=image, return_tensors="pt")
        return {name: tensor.squeeze(0) for name, tensor in inputs.items()}


def collate_inputs(batch):
    return {name: torch.stack([item[name] for item in batch]) for name in batch[0]}


def run_rtdetr_calibration(config):
    settings = config["calibration"]
    seed = int(settings.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(settings["device"])

    model_name = config["model"]["name"]
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device).eval()

    dataset_config = config["dataset"]
    base_dataset = CocoDetection(
        root=dataset_config["images"], annFile=dataset_config["annotations"]
    )
    sample_count = min(int(dataset_config["num_samples"]), len(base_dataset))
    dataset = ProcessedCocoDataset(
        Subset(base_dataset, range(sample_count)), processor
    )
    loader = DataLoader(
        dataset,
        batch_size=settings["collection_batch_size"],
        shuffle=False,
        collate_fn=collate_inputs,
    )

    decoder_layers = get_decoder_layers(model)
    layer_indices = parse_layers(config["linearization"]["layers"], len(decoder_layers))
    chunk_size = int(settings.get("chunk_size", 2))
    records = {}

    for start in range(0, len(layer_indices), chunk_size):
        chunk = layer_indices[start : start + chunk_size]
        with RTDetrActivationCollector(model, chunk) as collector:
            for batch in tqdm(loader, desc=f"Teacher activations {chunk}"):
                inputs = {name: tensor.to(device) for name, tensor in batch.items()}
                with torch.no_grad():
                    model(**inputs)

        for index in chunk:
            teacher_inputs, positions, teacher_outputs = collector.tensors(index)
            result = calibrate_rtdetr_layer(
                decoder_layers[index].self_attn,
                model.config,
                teacher_inputs,
                positions,
                teacher_outputs,
                settings,
            )
            records[index] = {
                "state_dict": result.module.state_dict(),
                "cosine_loss": result.cosine_loss,
                "mse_loss": result.mse_loss,
            }
            print(
                f"Layer {index}: cosine={result.cosine_loss:.6f}, "
                f"mse={result.mse_loss:.6f}"
            )
        del collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint = {
        "format_version": 1,
        "method": "LATTE",
        "model_name": model_name,
        "model_type": "rtdetr_decoder",
        "config": config,
        "layers": records,
    }
    output_path = Path(config["output"]["checkpoint_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved calibrated layers to {output_path}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Calibrate LATTE layers for RT-DETR")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_rtdetr_calibration(load_config(args.config))


if __name__ == "__main__":
    main()
