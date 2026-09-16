"""Load selected calibrated LATTE layers into pretrained RT-DETR."""

import argparse

import torch
from transformers import RTDetrForObjectDetection

from latte import replace_rtdetr_layers, select_layers_by_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--lambda-mse", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.layers is not None:
        layers = args.layers
    elif args.num_layers is not None:
        layers = select_layers_by_score(
            checkpoint, args.num_layers, mse_weight=args.lambda_mse
        )
    else:
        layers = sorted(int(index) for index in checkpoint["layers"])

    model = RTDetrForObjectDetection.from_pretrained(checkpoint["model_name"]).to(args.device)
    replace_rtdetr_layers(model, checkpoint, layers=layers, device=args.device)
    model.eval()
    print(f"Loaded LATTE into RT-DETR decoder layers: {layers}")


if __name__ == "__main__":
    main()
