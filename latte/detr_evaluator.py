"""COCO detection evaluation for calibrated DETR LATTE layers."""

import argparse
import json
import gc
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection
from transformers import DetrForObjectDetection, DetrImageProcessor

from .detr_utils import replace_detr_layers, select_layers_by_score, load_detr_checkpoint
from .detr_pipeline import load_config


def evaluate_model(model, processor, dataset, coco_gt, threshold, device):
    """Original DETR postprocessing, label mapping and COCO rounding."""
    category_ids = {
        cat["name"].lower(): cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }
    predictions = []
    image_ids = []
    for image, target in tqdm(dataset, desc="DETR evaluation"):
        if not target:
            continue
        image_id = int(target[0]["image_id"])
        image_ids.append(image_id)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)], device=device),
            threshold=threshold,
        )[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            # Preserve the original lookup; do not silently discard unknown labels.
            label_name = model.config.id2label[label.item()].lower()
            category_id = category_ids.get(label_name)
            if category_id is None:
                continue
            x0, y0, x1, y1 = box.tolist()
            predictions.append({
                "image_id": image_id, "category_id": category_id,
                "bbox": [round(x0, 2), round(y0, 2), round(x1-x0, 2), round(y1-y0, 2)],
                "score": round(score.item(), 3),
            })
    if not predictions:
        raise RuntimeError("DETR evaluation produced no predictions")
    with tempfile.TemporaryDirectory(prefix="latte_detr_") as directory:
        path = Path(directory) / "predictions.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(predictions, stream)
        detections = coco_gt.loadRes(str(path))
        evaluator = COCOeval(coco_gt, detections, iouType="bbox")
        # Full COCO evaluation matches the original; restrict only explicit subsets.
        if isinstance(dataset, Subset):
            evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def run_detr_evaluation(config):
    device = torch.device(config["evaluation"]["device"])
    checkpoint = load_detr_checkpoint(config["checkpoint"])
    model_name = checkpoint["model_name"]
    processor = DetrImageProcessor.from_pretrained(model_name)

    dataset_config = config["dataset"]
    dataset = CocoDetection(
        root=dataset_config["images"], annFile=dataset_config["annotations"]
    )
    if dataset_config.get("num_samples"):
        dataset = Subset(dataset, range(min(int(dataset_config["num_samples"]), len(dataset))))
    coco_gt = COCO(dataset_config["annotations"])
    output_path = Path(config["output"]["results_path"])
    results = []

    for count in config["evaluation"]["layer_counts"]:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        original_model = DetrForObjectDetection.from_pretrained(model_name).to(device)
        layers = select_layers_by_score(
            checkpoint,
            int(count),
            cosine_weight=float(config["selection"].get("cosine_weight", 1.0)),
            mse_weight=float(config["selection"].get("mse_weight", 10.0)),
        )
        model = deepcopy(original_model).to(device)
        replace_detr_layers(model, checkpoint, layers, device)
        model.eval()
        ap, ap50 = evaluate_model(
            model,
            processor,
            dataset,
            coco_gt,
            float(config["evaluation"].get("confidence_threshold", 0.001)),
            device,
        )
        record = {
            "num_layers": int(count),
            "linearized_layers": layers,
            "metrics": {
                "AP@[IoU=0.50:0.95]": round(ap, 3),
                "AP@[IoU=0.50]": round(ap50, 3),
            },
        }
        print(record)
        results.append(record)
        del model, original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR with LATTE layers")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_detr_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
