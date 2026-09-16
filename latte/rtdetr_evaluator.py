"""COCO detection evaluation for calibrated RT-DETR LATTE layers."""

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from .rtdetr_pipeline import load_config
from .rtdetr_utils import replace_rtdetr_layers, select_layers_by_score


def evaluate_model(model, processor, dataset, coco_gt, threshold, device):
    category_ids = {
        category["name"].lower(): category["id"]
        for category in coco_gt.loadCats(coco_gt.getCatIds())
    }
    predictions = []
    evaluated_image_ids = []
    for image, targets in tqdm(dataset, desc="RT-DETR evaluation"):
        if not targets:
            continue
        image_id = int(targets[0]["image_id"])
        evaluated_image_ids.append(image_id)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=device),
            threshold=threshold,
        )[0]
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            label_name = model.config.id2label[label.item()].lower()
            category_id = category_ids.get(label_name)
            if category_id is None:
                continue
            x0, y0, x1, y1 = box.tolist()
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score.item(),
                }
            )

    if not predictions:
        raise RuntimeError("Evaluation produced no COCO predictions")
    # COCO.loadRes expects a JSON path. Use a temporary file and remove it
    # automatically after this evaluation, retaining only the summary JSON.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as stream:
        json.dump(predictions, stream)
        stream.flush()
        coco_detections = coco_gt.loadRes(stream.name)
    evaluator = COCOeval(coco_gt, coco_detections, iouType="bbox")
    evaluator.params.imgIds = evaluated_image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def run_rtdetr_evaluation(config):
    device = torch.device(config["evaluation"]["device"])
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    original_model = RTDetrForObjectDetection.from_pretrained(model_name).eval()

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
        layers = select_layers_by_score(
            checkpoint,
            int(count),
            cosine_weight=float(config["selection"].get("cosine_weight", 1.0)),
            mse_weight=float(config["selection"].get("mse_weight", 10.0)),
        )
        model = deepcopy(original_model).to(device)
        replace_rtdetr_layers(model, checkpoint, layers, device)
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
                "AP@[IoU=0.50:0.95]": round(ap, 6),
                "AP@[IoU=0.50]": round(ap50, 6),
            },
        }
        print(record)
        results.append(record)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR with LATTE layers")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_rtdetr_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
