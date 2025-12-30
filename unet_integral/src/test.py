from pathlib import Path
from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig
import hydra
import json
import numpy as np
import pytorch_lightning as pl
import pycocotools.mask as mask_util

from src.modules.datamodule import SegmDataModule
from src.modules.lightning_module import SegmModule


def postprocess_results(result: list, out_dir: str, threshold: float = 0.5) -> None:
    """Postprocess segmentation results and save
    """
    print('Convert to rle and save to csv')
    rles = []
    im_names = []
    orig_sizes = []
    for entry in tqdm(result):
        preds, images_names, orig_size, infer_size = entry
        masks = (preds.squeeze(1).cpu().numpy() > threshold).astype(np.uint8)
        # rle = []
        for msk in masks:
            sample = mask_util.encode(np.asfortranarray(msk))
            sample['counts'] = sample['counts'].decode('ascii')
            rles.append(sample)
        # rles += rle
        im_names += images_names
        orig_sizes += orig_size

    result_df = pd.DataFrame(rles)
    result_df['orig_size'] = orig_sizes
    result_df['fname'] = im_names
    result_df.to_csv(out_dir / 'results.csv', index=False)


@hydra.main(config_path='../configs', config_name='config')
def test(config: DictConfig) -> None:
    """Inference function."""
    best_checkpoint_path = Path(
        # config.checkpoints) / config.experiment_name / f'best_{config.project_name}_{config.experiment_name}.ckpt'
        config.checkpoints) / config.experiment_name / f'best_segm_{config.experiment_name}.ckpt'


    model = SegmModule.load_from_checkpoint(best_checkpoint_path, config=config.model)
    trainer = pl.Trainer(
        devices=[config.device],
    )

    datamodule = SegmDataModule(config)
    # sample = next(iter(datamodule.test_dataloader()))
    # with open(Path(config.work_dir) / 'test_sample.pkl', 'wb') as f:
    #     pickle.dump(sample, f)

    # Test
    print(f'Test from {Path(config.data_path).name}.......')
    # datamodule.setup('test')
    # trainer.test(model, datamodule=datamodule)

    # Predict
    datamodule.setup('infer')
    result = trainer.predict(model, datamodule=datamodule)
    out_dir = Path(config.out_dir) / config.project_name / config.experiment_name / Path(config.data_path).parent.name
    out_dir.mkdir(exist_ok=True, parents=True)
    postprocess_results(result, out_dir)


if __name__ == '__main__':
    test()
