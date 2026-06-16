import typing as tp
from dataclasses import dataclass

from torch import nn
from omegaconf import DictConfig
from src.modules.io import load_object


@dataclass
class Loss:
    name: str
    head: str
    weight: float
    loss: nn.Module
    target_sigma: float = 2.0


def get_losses(losses_cfg: DictConfig) -> tp.List[Loss]:
    return [
        Loss(
            name=loss_cfg.name,
            head=getattr(loss_cfg, 'head', 'mask'),
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
            target_sigma=float(getattr(loss_cfg, 'target_sigma', 2.0)),
        ) for loss_cfg in losses_cfg
    ]
