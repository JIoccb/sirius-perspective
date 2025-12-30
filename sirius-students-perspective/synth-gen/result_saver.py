import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def save_composite(image, output_path: str, jpeg_quality: int) -> Path | None:
    path_obj = Path(output_path)
    success = cv2.imwrite(
        str(path_obj), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return path_obj if success else None


def build_records(
    filename: str,
    resolved_path: Path,
    width: int,
    height: int,
    keypoints_before: List[List[float]],
    keypoints_after: List[List[float]],
    dataset_name: str,
) -> Tuple[List[object], Dict[str, object]]:
    coords_str = json.dumps(keypoints_after, ensure_ascii=False)
    visibility = [1] * len(keypoints_after)
    visibility_str = json.dumps(visibility, ensure_ascii=False)
    row = [
        filename,
        str(resolved_path),
        width,
        height,
        coords_str,
        visibility_str,
        dataset_name,
    ]
    annotation = {
        "filename": filename,
        "filepath": str(resolved_path),
        "width": width,
        "height": height,
        "coords": keypoints_after,
        "visibility": visibility,
        "dataset": dataset_name,
    }
    return row, annotation
