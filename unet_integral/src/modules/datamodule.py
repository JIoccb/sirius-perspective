import logging
import os
import cv2
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

        self.train_transforms = get_train_transforms(width=self.config.dataset.image_width,
                                                     height=self.config.dataset.image_height,
                                                     )
        self.val_transforms = get_valid_transforms(width=self.config.dataset.image_width,
                                                   height=self.config.dataset.image_height,
                                                  )


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datamodule
        """

        if stage == 'fit':
            train_names, train_coords = read_df(self.data_path, 'train')
            valid_names, valid_coords = read_df(self.data_path, 'valid')

            self.train_dataset = SegmDataset(
                train_names,
                train_coords,
                transforms=self.train_transforms
            )
            self.valid_dataset = SegmDataset(
                valid_names,
                valid_coords,
                transforms=self.val_transforms
            )

        elif stage == 'test':
            test_names, test_coords = read_df(self.data_path, 'test')
            self.test_dataset = SegmDataset(
                test_names,
                test_coords,
                transforms=self.val_transforms
            )
            self.predict_dataset = SegmDataset(
                test_names,
                transforms=self.val_transforms
            )

        elif stage == 'infer':
            test_names = read_df(self.real_path)
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
        # print(mask_path.stem, image_path.stem)
        if mask_path.stem == image_path.stem:
            msk = cv2.imread(mask_path)
            img = cv2.imread(image_path)
            # print(msk.shape, img.shape)
            if (msk.shape[0] == img.shape[0]) and (msk.shape[1] == img.shape[1]):
                matches[i] = True
    return matches
            

def prepare_and_split_datasets(data_path: Path, train_fraction: float = 0.8, seed: int = 0) -> None:
    """Load raw dataset and prepare train, valid, test splits
    """

    # Load images list
    images_list = list(Path(data_path / 'images').glob("*.jpg"))
    masks_list = [Path(data_path / 'masks' / f"{path.stem}.png") for path in images_list]
    df = pd.DataFrame(np.array([images_list, masks_list]).T, columns=['images', 'masks'])
    # df = df.drop_duplicates()
    logging.info(f'Raw ds shape: {df.shape}')

    # Check image name equal mask name
    mask = compare_mask_image(df)
    df = df[mask]
    logging.info(f'DS shape after filter: {df.shape}')

    # Train/valid/test split
    np.random.seed(seed)
    indicies = np.arange(df.shape[0])
    np.random.shuffle(indicies)

    train_idx = indicies[:int(df.shape[0] * train_fraction)]
    valid_idx = indicies[train_idx.size:]
    test_idx = valid_idx[valid_idx.size // 2:]
    valid_idx = valid_idx[-test_idx.size:]

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


def read_df(data_path: Path, mode: str = None) -> Tuple[List, List]:
    """
    Read df with annotations to list of image paths and coords list
    """
    
    if mode is not None:
        df = pd.read_csv(data_path / f'df_{mode}.csv')
        image_names = df['images'].to_list()
        mask_names = df['masks'].to_list()

        return image_names, mask_names
    
    else:

        df = pd.read_csv(data_path)
        image_names = df['filepath'].apply(lambda x: Path(data_path).parent / Path(x).parent.parent.name / Path(x).parent.name / Path(x).name).to_list()
        
        return image_names




