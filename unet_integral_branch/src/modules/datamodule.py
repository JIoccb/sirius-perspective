import logging
import os
import cv2
import json
from typing import Optional, Tuple, List
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch

from src.modules.transforms import get_train_transforms, get_valid_transforms
from src.modules.dataset import SegmDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class SegmDataModule(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.data_path = Path(config.data_path)
        self.real_path = Path(config.real_path) if config.get('real_path') is not None else None

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Enable keypoints pipeline if the model config requests it or losses include a non-mask head
        model_cfg = getattr(config, 'model', {})
        self.with_keypoints = bool(getattr(model_cfg, 'with_keypoints', False))
        try:
            if not self.with_keypoints and hasattr(model_cfg, 'losses'):
                self.with_keypoints = any(getattr(l, 'head', 'mask') != 'mask' for l in model_cfg.losses)
        except Exception:
            pass

        self.train_transforms = get_train_transforms(
            width=self.config.dataset.image_width,
            height=self.config.dataset.image_height,
            with_keypoints=self.with_keypoints,
        )
        self.val_transforms = get_valid_transforms(
            width=self.config.dataset.image_width,
            height=self.config.dataset.image_height,
            with_keypoints=self.with_keypoints,
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datamodule
        """

        if stage == 'fit':
            train_names, train_masks, train_keypoints, train_valid = read_df(self.data_path, 'train')
            valid_names, valid_masks, valid_keypoints, valid_valid = read_df(self.data_path, 'valid')

            self.train_dataset = SegmDataset(
                train_names,
                train_masks,
                keypoints=train_keypoints,
                keypoints_valid=train_valid,
                transforms=self.train_transforms,
                with_keypoints=self.with_keypoints,
            )
            self.valid_dataset = SegmDataset(
                valid_names,
                valid_masks,
                keypoints=valid_keypoints,
                keypoints_valid=valid_valid,
                transforms=self.val_transforms,
                with_keypoints=self.with_keypoints,
            )

        elif stage == 'test':
            test_names, test_masks, test_keypoints, test_valid = read_df(self.data_path, 'test')
            self.test_dataset = SegmDataset(
                test_names,
                test_masks,
                keypoints=test_keypoints,
                keypoints_valid=test_valid,
                transforms=self.val_transforms,
                with_keypoints=self.with_keypoints,
            )
            self.predict_dataset = SegmDataset(
                test_names,
                transforms=self.val_transforms
            )

        elif stage == 'infer':
            test_names = read_infer_paths(self.real_path)
            self.predict_dataset = SegmDataset(
                test_names,
                transforms=self.val_transforms
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )


def compare_mask_image(df):

    matches = [False] * df.shape[0]
    for i in range(df.shape[0]):
        mask_path = Path(df['masks'].iloc[i])
        image_path = Path(df['images'].iloc[i])
        if mask_path.stem == image_path.stem:
            msk = cv2.imread(str(mask_path))
            img = cv2.imread(str(image_path))
            if msk is None or img is None:
                continue
            if (msk.shape[0] == img.shape[0]) and (msk.shape[1] == img.shape[1]):
                matches[i] = True
    return matches
            

def prepare_and_split_datasets(data_path: Path, train_fraction: float = 0.8, seed: int = 0) -> None:
    """Load raw dataset and prepare train, valid, test splits
    """

    data_path = Path(data_path)
    annotations_path = data_path / 'annotations.csv'

    if annotations_path.exists():
        df = read_annotations_df(annotations_path)
    else:
        images_list = list(Path(data_path / 'images').glob("*.jpg"))
        masks_list = [Path(data_path / 'masks' / f"{path.stem}.png") for path in images_list]
        df = pd.DataFrame(np.array([images_list, masks_list]).T, columns=['images', 'masks'])

    logging.info(f'Raw ds shape: {df.shape}')

    mask = compare_mask_image(df)
    df = df[mask]
    logging.info(f'DS shape after filter: {df.shape}')

    np.random.seed(seed)
    indicies = np.arange(df.shape[0])
    np.random.shuffle(indicies)

    train_end = int(df.shape[0] * train_fraction)
    train_idx = indicies[:train_end]
    rest = indicies[train_end:]
    mid = rest.size // 2
    valid_idx = rest[:mid]
    test_idx = rest[mid:]

    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]
    test_df = df.iloc[test_idx]

    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')


    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')


def _resolve_path(path_value: str, base_dir: Path) -> str:
    pth = Path(path_value)
    if pth.is_absolute():
        return str(pth)
    candidate = base_dir / pth
    if candidate.exists():
        return str(candidate)
    return str(base_dir.parent / pth)


def _parse_keypoints(value) -> List[List[float]]:
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float32).reshape(4, 2)
    return arr.tolist()


def _parse_visibility(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 1.0
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr.min()) if arr.size else 1.0


def read_annotations_df(annotations_path: Path) -> pd.DataFrame:
    """Read generator annotations and normalize them to training columns."""
    annotations_path = Path(annotations_path)
    df = pd.read_csv(annotations_path)
    if 'filepath' not in df.columns:
        raise ValueError(f"Expected 'filepath' column in {annotations_path}")

    csv_dir = annotations_path.parent
    image_names = df['filepath'].apply(lambda value: _resolve_path(str(value), csv_dir))
    mask_names = image_names.apply(lambda value: str(csv_dir / 'masks' / f'{Path(value).stem}.png'))

    out = pd.DataFrame({'images': image_names, 'masks': mask_names})
    if 'coords' in df.columns:
        out['keypoints'] = df['coords'].apply(lambda value: json.dumps(_parse_keypoints(value)))
    if 'visibility' in df.columns:
        out['keypoints_valid'] = df['visibility'].apply(_parse_visibility)
    elif 'coords' in df.columns:
        out['keypoints_valid'] = 1.0
    return out


def read_df(data_path: Path, mode: str) -> Tuple[List, List, Optional[List], Optional[List]]:
    """Read split dataframe to image, mask and optional keypoint lists."""
    df_path = Path(data_path) / f'df_{mode}.csv'
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset split does not exist: {df_path}")

    df = pd.read_csv(df_path)
    image_names = df['images'].to_list()
    mask_names = df['masks'].to_list()
    keypoints = None
    keypoints_valid = None

    if 'keypoints' in df.columns:
        keypoints = df['keypoints'].apply(_parse_keypoints).to_list()
        keypoints_valid = (
            df['keypoints_valid'].astype(float).to_list()
            if 'keypoints_valid' in df.columns
            else [1.0] * len(keypoints)
        )

    return image_names, mask_names, keypoints, keypoints_valid


def read_infer_paths(annotations_path: Path) -> List[str]:
    """Read image paths for inference from an annotations CSV."""
    annotations_path = Path(annotations_path)
    df = pd.read_csv(annotations_path)
    if 'filepath' not in df.columns:
        raise ValueError(f"Expected 'filepath' column in {annotations_path}")
    csv_dir = annotations_path.parent

    image_names = []
    for _, row in df.iterrows():
        resolved = Path(_resolve_path(str(row['filepath']), csv_dir))
        if resolved.exists():
            image_names.append(str(resolved))
            continue

        if 'dataset' in row and 'filename' in row:
            local_candidate = csv_dir / str(row['dataset']) / str(row['filename'])
            if local_candidate.exists():
                image_names.append(str(local_candidate))
                continue
            local_candidate = csv_dir / str(row['dataset']) / 'images' / str(row['filename'])
            if local_candidate.exists():
                image_names.append(str(local_candidate))
                continue

        image_names.append(str(resolved))
    return image_names

