import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
import cv2

from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Generate submission.csv using DSGCNet", add_help=True
    )
    parser.add_argument(
        "--backbone",
        default="vgg16_bn",
        type=str,
        help="backbone name: vgg16 or vgg16_bn",
    )
    parser.add_argument(
        "--row", default=2, type=int, help="row number of anchor points"
    )
    parser.add_argument(
        "--line", default=2, type=int, help="line number of anchor points"
    )
    parser.add_argument(
        "--backbone_pretrained",
        default=0,
        type=int,
        help="set 0 to avoid downloading pretrained weights (useful in offline Kaggle)",
    )
    parser.add_argument(
        "--use_gcn",
        default=0,
        type=int,
        help="set 0 to disable GCN if torch_geometric is unavailable",
    )
    parser.add_argument(
        "--weight_path",
        default="./weights/SHTechA.pth",
        type=str,
        help='path to trained checkpoint (.pth) containing key "model"',
    )
    parser.add_argument(
        "--test_dir",
        default="./crowd_datasets/HOLOGY/test",
        type=str,
        help="directory containing test images (recursively scanned for .jpg/.png/.jpeg)",
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float, help="score threshold for valid points"
    )
    parser.add_argument(
        "--gpu_id", default=0, type=int, help="GPU id to use if available"
    )
    parser.add_argument(
        "--output_csv", default="submission.csv", type=str, help="output CSV path"
    )
    return parser


def load_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    model.to(device)
    if args.weight_path and os.path.exists(args.weight_path):
        ckpt = torch.load(args.weight_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded weights: {args.weight_path}")
    else:
        print(
            f"[Warning] Weight file not found: {args.weight_path}. Using randomly initialized model."
        )
    model.eval()
    return model, device


def build_transform():
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def iter_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@torch.no_grad()
def predict_count(model, device, transform, img_path: Path, threshold: float) -> int:
    img_raw = Image.open(img_path).convert("RGB")
    w, h = img_raw.size
    new_w = (w // 128) * 128 if w >= 128 else 128
    new_h = (h // 128) * 128 if h >= 128 else 128
    if (new_w, new_h) != (w, h):
        img_raw = img_raw.resize((new_w, new_h), cv2.INTER_CUBIC)
    img = transform(img_raw)
    samples = torch.tensor(img).unsqueeze(0).to(device)
    outputs = model(samples)
    scores = torch.nn.functional.softmax(outputs["pred_logits"], dim=-1)[:, :, 1][0]
    valid = scores > threshold
    return int(valid.sum().item())


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    model, device = load_model(args)
    transform = build_transform()

    records = []
    for img_path in sorted(iter_images(args.test_dir)):
        count = predict_count(model, device, transform, img_path, args.threshold)
        records.append({"image_id": img_path.name, "count": count})

    if not records:
        print(f"[Warning] No images found under {args.test_dir}")
    df = pd.DataFrame.from_records(records, columns=["image_id", "count"])
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(records)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
