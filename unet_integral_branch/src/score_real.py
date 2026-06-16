from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.modules.lightning_module import SegmModule
from src.modules.transforms import get_valid_transforms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score keypoint predictions on annotated real-test CSV.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--gt-order",
        choices=["tl_tr_br_bl", "tl_tr_bl_br"],
        default="tl_tr_bl_br",
        help="Corner order in the annotation CSV.",
    )
    return parser


def resolve_image_path(row: pd.Series, csv_dir: Path) -> Path:
    path = Path(str(row["filepath"]))
    if path.exists():
        return path
    if "dataset" in row and "filename" in row:
        candidate = csv_dir / str(row["dataset"]) / str(row["filename"])
        if candidate.exists():
            return candidate
        candidate = csv_dir / str(row["dataset"]) / "images" / str(row["filename"])
        if candidate.exists():
            return candidate
    return path


def parse_keypoints(value: str, order: str) -> np.ndarray:
    keypoints = np.asarray(json.loads(value), dtype=np.float32).reshape(4, 2)
    if order == "tl_tr_bl_br":
        keypoints = keypoints[[0, 1, 3, 2]]
    return keypoints


def predict_keypoints(model: SegmModule, image_path: Path, transforms, device: torch.device) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transforms(image=image_rgb)
    tensor = transformed["image"].float().unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor)
    return preds["keypoints"].detach().cpu().numpy().reshape(4, 2)


def main() -> None:
    args = build_parser().parse_args()
    config = OmegaConf.load(args.config)
    annotations_path = args.annotations or Path(config.real_path)
    checkpoint = args.checkpoint or Path(config.checkpoints) / config.experiment_name / f"best_segm_{config.experiment_name}.ckpt"
    out_path = args.out or Path(config.out_dir) / config.project_name / config.experiment_name / "real_test_scores.csv"

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() and config.accelerator == "gpu" else "cpu")
    model = SegmModule.load_from_checkpoint(checkpoint, config=config.model, map_location=device)
    model.to(device)
    model.eval()

    transforms = get_valid_transforms(
        width=config.dataset.image_width,
        height=config.dataset.image_height,
        with_keypoints=False,
    )

    df = pd.read_csv(annotations_path)
    csv_dir = Path(annotations_path).parent
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        image_path = resolve_image_path(row, csv_dir)
        gt = parse_keypoints(row["coords"], args.gt_order)
        pred = predict_keypoints(model, image_path, transforms, device)

        width = float(row["width"])
        height = float(row["height"])
        scale = np.array([width, height], dtype=np.float32)
        abs_error = np.abs(pred - gt) * scale
        euclidean = np.linalg.norm((pred - gt) * scale, axis=1)

        rows.append({
            "filename": row.get("filename", image_path.name),
            "dataset": row.get("dataset", ""),
            "filepath": str(image_path),
            "mae_px": float(abs_error.mean()),
            "mean_euclidean_px": float(euclidean.mean()),
            "mape_percent": float((np.abs(pred - gt) / np.maximum(gt, 1e-6)).mean() * 100.0),
            **{f"pred_x{i + 1}": float(pred[i, 0]) for i in range(4)},
            **{f"pred_y{i + 1}": float(pred[i, 1]) for i in range(4)},
            **{f"gt_x{i + 1}": float(gt[i, 0]) for i in range(4)},
            **{f"gt_y{i + 1}": float(gt[i, 1]) for i in range(4)},
        })

    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Saved per-image scores to {out_path}")
    print(f"MAE px: {result['mae_px'].mean():.3f}")
    print(f"Mean euclidean px: {result['mean_euclidean_px'].mean():.3f}")
    print(f"MAPE %: {result['mape_percent'].mean():.3f}")


if __name__ == "__main__":
    main()
