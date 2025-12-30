import typing as tp
from dataclasses import dataclass

from torch import nn
from omegaconf import DictConfig
from src.modules.io import load_object


@dataclass
class Loss:
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: DictConfig) -> tp.List[Loss]:
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        ) for loss_cfg in losses_cfg
    ]
