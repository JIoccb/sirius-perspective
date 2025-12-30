"""Training pipeline.
"""

import hydra
import shutil
import pickle
from omegaconf import DictConfig
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.modules.datamodule import SegmDataModule, prepare_and_split_datasets
from src.modules.lightning_module import SegmModule


@hydra.main(config_path='../configs', config_name='config')
def main(config: DictConfig) -> None:
    """Training function."""
    work_dir = Path(config.work_dir)
    experiments_path = work_dir / 'experiments' / config.project_name
    experiment_save_path = experiments_path / config.experiment_name
    experiment_save_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = Path(config.checkpoints) / config.experiment_name
    best_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Set logger
    log_dir = experiments_path / 'log_dir'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(
        log_dir,
        name=config.experiment_name
    )

    pl.seed_everything(config.seed, workers=True)

    datamodule = SegmDataModule(config)
    prepare_and_split_datasets(Path(config.data_path),
                                          config.dataset.train_size,
                                          config.seed
                                          )
    datamodule.setup('fit')
    sample = next(iter(datamodule.train_dataloader()))
    with open(work_dir / 'sample.pkl', 'wb') as f:
        pickle.dump(sample, f)

    model = SegmModule(config.model)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.model.checkpoint_callback.monitor_metric,
        mode=config.model.checkpoint_callback.monitor_mode,
        filename=f'epoch_{{epoch:02d}}-{{{config.model.checkpoint_callback.monitor_metric}:.3f}}',
        verbose=config.model.checkpoint_callback.verbose
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_nsteps,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.model.checkpoint_callback.monitor_metric, patience=10,
                          mode=config.model.checkpoint_callback.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Copy best checkpoint to checkpoint_path
    best_checkpoint_path = str(best_checkpoint_path / f'best_segm_{config.experiment_name}.ckpt')
    shutil.copyfile(checkpoint_callback.best_model_path, best_checkpoint_path)


if __name__ == '__main__':
    main()
