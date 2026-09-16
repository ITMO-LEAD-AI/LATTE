"""Load selected calibrated LATTE layers into pretrained DETR."""

import argparse

import torch
from transformers import DetrForObjectDetection

from latte.detr_utils import replace_detr_layers, select_layers_by_score, load_detr_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--lambda-mse", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = load_detr_checkpoint(args.checkpoint)
    if args.layers is not None:
        layers = args.layers
    elif args.num_layers is not None:
        layers = select_layers_by_score(
            checkpoint, args.num_layers, mse_weight=args.lambda_mse
        )
    else:
        layers = sorted(int(index) for index in checkpoint["layers"])

    model = DetrForObjectDetection.from_pretrained(checkpoint["model_name"]).to(args.device)
    replace_detr_layers(model, checkpoint, layers=layers, device=args.device)
    model.eval()
    print(f"Loaded LATTE into DETR decoder layers: {layers}")


if __name__ == "__main__":
    main()
