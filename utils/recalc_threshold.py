import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

import main as m
from model.MyDataset import MyTorchDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute and save anomaly threshold for a trained model.")
    parser.add_argument("--service_s", required=True, help="Service name (e.g., mobservice2)")
    parser.add_argument("--model_path", required=True, help="Path to trained checkpoint .pkl")
    parser.add_argument("--output", help="Where to save threshold.json (default: alongside model)")
    parser.add_argument("--window_size", type=int, default=20, help="Window size used during training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for calibration loader")
    parser.add_argument("--quantile", type=float, default=0.99, help="Quantile for threshold selection")
    parser.add_argument("--proportion", type=float, default=0.6, help="Train proportion for get_data (matches training)")
    return parser.parse_args()


def main():
    args = parse_args()

    # main.py expects global m.args with service name
    m.args = SimpleNamespace(service_s=args.service_s)

    print("Loading train data...")
    _, label_with_timestamp, channels, nmi_matrix = m.get_data(
        "train", f"{args.service_s}train_nmiMatrix.pk", args.proportion
    )

    train_data = MyTorchDataset(
        label_with_timestamp=label_with_timestamp,
        channels=channels,
        aj_matrix=nmi_matrix,
        window_size=args.window_size,
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_path = Path(args.model_path)
    print(f"Loading model from {model_path} ...")
    net = torch.load(model_path, weights_only=False).to(m.cuda_device).eval()
    print("Model loaded.")

    print(f"Calibrating threshold (quantile={args.quantile})...")
    threshold = m.calibrate_threshold(train_loader, net, quantile=args.quantile)
    if threshold is None:
        raise SystemExit("No threshold computed (no scores collected). Try smaller batch size.")

    output_path = Path(args.output) if args.output else model_path.parent / "threshold.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"threshold": float(threshold), "quantile": args.quantile}, f)
    print(f"Saved threshold {threshold:.6f} to {output_path}")


if __name__ == "__main__":
    main()
