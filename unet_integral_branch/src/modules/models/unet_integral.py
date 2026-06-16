from __future__ import annotations

import inspect
import torch
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


def softargmax_2d(heatmaps: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Integral (soft-argmax) regression from heatmaps.

    Args:
        heatmaps: (N,K,H,W) logits.

    Returns:
        coords: (N,K,2) in [0,1] (x,y) relative to W,H.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected (N,K,H,W), got {tuple(heatmaps.shape)}")
    n, k, h, w = heatmaps.shape

    probs = F.softmax(heatmaps.view(n, k, -1), dim=-1)  # (N,K,HW)

    # coordinate grid in [0,1]
    ys = torch.linspace(0.0, 1.0, steps=h, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.linspace(0.0, 1.0, steps=w, device=heatmaps.device, dtype=heatmaps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
    xx = xx.reshape(-1)  # (HW)
    yy = yy.reshape(-1)

    exp_x = (probs * xx[None, None, :]).sum(dim=-1)
    exp_y = (probs * yy[None, None, :]).sum(dim=-1)
    coords = torch.stack([exp_x, exp_y], dim=-1)  # (N,K,2)
    return coords


class UnetIntegral(nn.Module):
    """UNet mask head + integral-regression keypoints head.

    Outputs a dict:
      - 'mask':     (N, C, H, W) logits
      - 'keypoints': (N, 2K) normalized coords in [0,1]
      - 'heatmaps': (N, K, H, W) keypoint heatmap logits (optional use)
    """

    def __init__(
        self,
        encoder: str = 'resnet34',
        encoder_weights: str | None = 'imagenet',
        in_channels: int = 3,
        mask_classes: int = 1,
        num_keypoints: int = 4,
        keypoint_head_channels: int = 32,
        heatmap_temperature: float = 1.0,
        **unet_kwargs,
    ) -> None:
        super().__init__()

        # Base UNet without caring about its forward()
        self.encoder = smp.encoders.get_encoder(
            name=encoder,
            in_channels=in_channels,
            depth=unet_kwargs.get('encoder_depth', 5),
            weights=encoder_weights,
        )

        decoder_channels = unet_kwargs.get('decoder_channels', (256, 128, 64, 32, 16))
        decoder_depth = unet_kwargs.get('encoder_depth', 5)
        use_batchnorm = unet_kwargs.get('decoder_use_batchnorm', True)
        decoder_kwargs = {
            'encoder_channels': self.encoder.out_channels,
            'decoder_channels': decoder_channels,
            'n_blocks': decoder_depth,
            'attention_type': unet_kwargs.get('attention_type', None),
        }
        decoder_params = inspect.signature(smp.decoders.unet.decoder.UnetDecoder).parameters
        if 'use_batchnorm' in decoder_params:
            decoder_kwargs['use_batchnorm'] = use_batchnorm
        if 'use_norm' in decoder_params:
            decoder_kwargs['use_norm'] = 'batchnorm' if use_batchnorm else None
        if 'center' in decoder_params:
            decoder_kwargs['center'] = unet_kwargs.get('center', False)
        if 'add_center_block' in decoder_params:
            decoder_kwargs['add_center_block'] = unet_kwargs.get('center', False)
        if 'interpolation_mode' in decoder_params:
            decoder_kwargs['interpolation_mode'] = unet_kwargs.get('interpolation_mode', 'nearest')
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(**decoder_kwargs)

        last_ch = decoder_channels[-1]
        self.mask_head = smp.base.SegmentationHead(
            in_channels=last_ch,
            out_channels=mask_classes,
            activation=None,
            kernel_size=3,
        )

        self.kp_head = nn.Sequential(
            nn.Conv2d(last_ch, keypoint_head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(keypoint_head_channels, num_keypoints, kernel_size=1),
        )
        self.num_keypoints = int(num_keypoints)
        self.heatmap_temperature = float(heatmap_temperature)

        self.initialize()

    def initialize(self) -> None:
        smp.base.initialization.initialize_decoder(self.decoder)
        smp.base.initialization.initialize_head(self.mask_head)
        for module in self.kp_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        try:
            dec = self.decoder(*feats)
        except TypeError:
            dec = self.decoder(feats)

        mask_logits = self.mask_head(dec)
        heatmap_logits = self.kp_head(dec)
        if self.heatmap_temperature != 1.0:
            heatmap_logits = heatmap_logits / self.heatmap_temperature

        coords = softargmax_2d(heatmap_logits)  # (N,K,2) in [0,1]
        coords_flat = coords.view(coords.shape[0], -1)  # (N,2K)

        return {
            'mask': mask_logits,
            'keypoints': coords_flat,
            'heatmaps': heatmap_logits,
        }
