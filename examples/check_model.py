"""Quick check: build PlantXMamba, count parameters and run one forward pass (random weights)."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plantxmamba import PlantXMamba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=38)
    args = ap.parse_args()

    model = PlantXMamba(num_classes=args.num_classes, pretrained=False).eval()
    params = sum(p.numel() for p in model.parameters())
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 224, 224))

    print(f"classes: {args.num_classes}")
    print(f"parameters: {params:,} ({params / 1e6:.2f}M, {params * 4 / 2**20:.2f} MB in float32)")
    print(f"output: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
