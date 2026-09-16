"""Print the calibration scores stored in a LATTE checkpoint."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Model: {checkpoint['model_name']}")
    print("layer\tcosine_loss\tmse_loss")
    for index, record in sorted(checkpoint["layers"].items(), key=lambda item: int(item[0])):
        print(f"{index}\t{record['cosine_loss']:.6f}\t{record['mse_loss']:.6f}")


if __name__ == "__main__":
    main()
