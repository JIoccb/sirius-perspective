import cv2
import numpy as np
from typing import List, Tuple, Optional, Sequence

import torch
from torch.utils.data import Dataset

from src.modules.io import read_rgb_img


def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2) points, got {pts.shape}")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _corners_from_mask(mask01: np.ndarray) -> Tuple[np.ndarray, float]:
    """Derive 4 corners from a binary mask.

    Returns:
        corners_xy: (4,2) float32 in pixel coords.
        valid: 1.0 if corners found, else 0.0.
    """
    m = (mask01.squeeze() > 0.5).astype(np.uint8) * 255
    if m.ndim != 2:
        m = m[..., 0]

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((4, 2), dtype=np.float32), 0.0

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return np.zeros((4, 2), dtype=np.float32), 0.0

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # (4,2)
    box = _order_points_tl_tr_br_bl(box)
    return box.astype(np.float32), 1.0


def _normalized_to_pixel_keypoints(
    keypoints_norm: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalized (x,y) keypoints to pixel coords for Albumentations."""
    kps = np.asarray(keypoints_norm, dtype=np.float32).reshape(4, 2)
    scale = np.array([width, height], dtype=np.float32)
    kps_xy = kps * scale
    kps_xy[:, 0] = np.clip(kps_xy[:, 0], 0, max(width - 1, 0))
    kps_xy[:, 1] = np.clip(kps_xy[:, 1], 0, max(height - 1, 0))
    return kps_xy


class SegmDataset(Dataset):
    """Dataset for segmentation or segmentation+keypoints.

    Output formats:
      - Train/val (segmentation only): (image, image_path, (H,W), mask)
      - Train/val (with keypoints):    (image, image_path, (H,W), mask, keypoints_norm, kps_valid)
      - Infer:                         (image, image_path, (H,W), image.shape[1:])

    keypoints_norm is (4,2) in [0,1] relative to *inference* size after transforms.
    """

    def __init__(
        self,
        images_names: List,
        masks_names: Optional[List] = None,
        keypoints: Optional[List] = None,
        keypoints_valid: Optional[List] = None,
        transforms=None,
        with_keypoints: bool = False,
    ):
        super().__init__()
        self.images_names = images_names
        self.masks_names = masks_names
        self.keypoints = keypoints
        self.keypoints_valid = keypoints_valid
        self.transforms = transforms
        self.with_keypoints = with_keypoints

    def __getitem__(self, idx: int):

        # Read image
        image_path = self.images_names[idx]
        image = read_rgb_img(str(image_path))
        height, width = image.shape[:2]

        # Train/val branch
        if self.masks_names is not None:
            mask = cv2.imread(str(self.masks_names[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask does not exist: {self.masks_names[idx]}")
            mask = (mask[..., None] / 255.0).astype(np.float32)

            if self.with_keypoints:
                if self.keypoints is not None:
                    kps_xy = _normalized_to_pixel_keypoints(self.keypoints[idx], width, height)
                    kps_valid = float(self.keypoints_valid[idx]) if self.keypoints_valid is not None else 1.0
                else:
                    kps_xy, kps_valid = _corners_from_mask(mask)

                transformed = self.transforms(image=image, mask=mask, keypoints=kps_xy.tolist())
                image_t = transformed['image'].float()
                mask_t = transformed['mask'].float()

                # Albumentations returns list[tuple(x,y)]
                kps_xy_t = np.array(transformed['keypoints'], dtype=np.float32).reshape(4, 2)
                h_inf, w_inf = int(image_t.shape[1]), int(image_t.shape[2])
                denom = np.array([max(w_inf - 1, 1), max(h_inf - 1, 1)], dtype=np.float32)
                kps_norm = (kps_xy_t / denom).clip(0.0, 1.0)

                return (
                    image_t,
                    str(image_path),
                    (height, width),
                    mask_t,
                    torch.from_numpy(kps_norm),
                    torch.tensor(kps_valid, dtype=torch.float32),
                )

            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image'].float()
            mask = transformed['mask'].float()
            return image, str(image_path), (height, width), mask

        # Infer branch
        transformed = self.transforms(image=image)
        image = transformed['image'].float()
        return image, str(image_path), (height, width), image.shape[1:]

    def __len__(self) -> int:
        return len(self.images_names)
