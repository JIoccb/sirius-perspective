import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from src.modules.io import load_object
from src.modules.losses import get_losses
from src.modules.metrics import get_mask_metrics, get_kps_metrics


def make_gaussian_heatmaps(
    keypoints: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build normalized Gaussian heatmaps from keypoints in [0, 1]."""
    if keypoints.ndim == 2:
        keypoints = keypoints.view(keypoints.shape[0], -1, 2)

    ys = torch.arange(height, device=keypoints.device, dtype=keypoints.dtype)
    xs = torch.arange(width, device=keypoints.device, dtype=keypoints.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    px = keypoints[..., 0].clamp(0.0, 1.0) * max(width - 1, 1)
    py = keypoints[..., 1].clamp(0.0, 1.0) * max(height - 1, 1)

    dist2 = (xx[None, None] - px[..., None, None]) ** 2 + (yy[None, None] - py[..., None, None]) ** 2
    heatmaps = torch.exp(-dist2 / (2.0 * sigma * sigma))
    if valid is not None:
        heatmaps = heatmaps * valid.view(-1, 1, 1, 1)
    return heatmaps


def keypoint_mae_px(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    image_size: tuple[int, int],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute keypoint error in pixels for tensors in [0, 1]."""
    height, width = image_size
    scale = pred_keypoints.new_tensor([width, height]).repeat(pred_keypoints.shape[1] // 2)
    error = torch.abs(pred_keypoints - gt_keypoints) * scale
    if valid is not None:
        error = error * valid
        denom = valid.sum().clamp_min(1.0) * pred_keypoints.shape[1]
        return error.sum() / denom
    return error.mean()


class SegmModule(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # Init model (supports both SMP classes and dotted-path custom models)
        if isinstance(self.config.model_arch, str) and '.' in self.config.model_arch:
            Net = load_object(self.config.model_arch)
        else:
            Net = getattr(smp, self.config.model_arch)
        self.model = Net(**self.config.model_args)

        # Metrics
        thr = 0.5
        if hasattr(self.config, 'metrics_kwargs') and self.config.metrics_kwargs is not None:
            thr = float(getattr(self.config.metrics_kwargs, 'threshold', thr))
        self.val_mask_metrics = get_mask_metrics(threshold=thr).clone(prefix='val_')
        self.test_mask_metrics = get_mask_metrics(threshold=thr).clone(prefix='test_')
        self.val_kps_metrics = get_kps_metrics().clone(prefix='val_')
        self.test_kps_metrics = get_kps_metrics().clone(prefix='test_')

        # Set losses
        self.losses = get_losses(self.config.losses)

        # Save hparams
        self.save_hyperparameters(dict(self.config))

    def configure_optimizers(self):
        optimizer = load_object(self.config.optimizer)(
            self.model.parameters(), **self.config.optimizer_kwargs,
        )
        scheduler = load_object(self.config.scheduler)(optimizer, **self.config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.config.checkpoint_callback.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, images_names, im_size, *rest = batch
        images = torch.stack(images)

        gt_mask = torch.stack(rest[0]) if len(rest) > 0 else None
        gt_kps = torch.stack(rest[1]) if len(rest) > 1 else None
        kps_valid = torch.stack(rest[2]).view(-1, 1) if len(rest) > 2 else None

        preds = self(images)
        loss = self.calculate_loss(preds, gt_mask, gt_kps, kps_valid, 'train_')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, images_names, im_size, *rest = batch
        images = torch.stack(images)

        gt_mask = torch.stack(rest[0]) if len(rest) > 0 else None
        gt_kps = torch.stack(rest[1]) if len(rest) > 1 else None
        kps_valid = torch.stack(rest[2]).view(-1, 1) if len(rest) > 2 else None

        preds = self(images)
        loss = self.calculate_loss(preds, gt_mask, gt_kps, kps_valid, 'val_')

        # Mask metrics on probabilities
        if gt_mask is not None:
            mask_logits = preds['mask'] if isinstance(preds, dict) else preds
            mask_prob = torch.sigmoid(mask_logits)
            m = self.val_mask_metrics(mask_prob, gt_mask)
            self.log('val_f1', m['val_f1'], on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_iou', m['val_iou'], on_step=False, on_epoch=True, prog_bar=True)

        # Keypoints metrics
        if gt_kps is not None and isinstance(preds, dict) and 'keypoints' in preds:
            pred_kps = preds['keypoints']
            gt_kps_flat = gt_kps.view(gt_kps.shape[0], -1)
            if kps_valid is not None:
                pred_kps = pred_kps * kps_valid
                gt_kps_flat = gt_kps_flat * kps_valid
            self.val_kps_metrics(pred_kps, gt_kps_flat)
            self.log(
                'val_mae_px',
                keypoint_mae_px(pred_kps, gt_kps_flat, images.shape[-2:], kps_valid),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, images_names, im_size, *rest = batch
        images = torch.stack(images)

        gt_mask = torch.stack(rest[0]) if len(rest) > 0 else None
        gt_kps = torch.stack(rest[1]) if len(rest) > 1 else None
        kps_valid = torch.stack(rest[2]).view(-1, 1) if len(rest) > 2 else None

        preds = self(images)
        if gt_mask is not None:
            mask_logits = preds['mask'] if isinstance(preds, dict) else preds
            mask_prob = torch.sigmoid(mask_logits)
            self.test_mask_metrics(mask_prob, gt_mask)
        if gt_kps is not None and isinstance(preds, dict) and 'keypoints' in preds:
            pred_kps = preds['keypoints']
            gt_kps_flat = gt_kps.view(gt_kps.shape[0], -1)
            if kps_valid is not None:
                pred_kps = pred_kps * kps_valid
                gt_kps_flat = gt_kps_flat * kps_valid
            self.test_kps_metrics(pred_kps, gt_kps_flat)

    def predict_step(self, batch, batch_idx):
        images, images_names, orig_size, infer_size = batch
        images = torch.stack(images)
        preds = self(images)

        # Always return mask probabilities as first element for backward compatibility
        if isinstance(preds, dict):
            mask_prob = torch.sigmoid(preds['mask'])
            return mask_prob, preds.get('keypoints', None), images_names, orig_size, infer_size
        else:
            mask_prob = torch.sigmoid(preds)
            return mask_prob, None, images_names, orig_size, infer_size


    def on_validation_epoch_start(self) -> None:
        self.val_mask_metrics.reset()
        self.val_kps_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_mask_metrics.compute(), on_epoch=True, on_step=False)
        # Only log if ever updated
        try:
            self.log_dict(self.val_kps_metrics.compute(), on_epoch=True, on_step=False)
        except Exception:
            pass

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_mask_metrics.compute(), on_epoch=True, on_step=False)
        try:
            self.log_dict(self.test_kps_metrics.compute(), on_epoch=True, on_step=False)
        except Exception:
            pass


    def calculate_loss(
        self,
        preds: torch.Tensor,
        gt_masks: torch.Tensor | None,
        gt_keypoints: torch.Tensor | None,
        kps_valid: torch.Tensor | None,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for _loss in self.losses:
            if _loss.head == 'mask':
                if gt_masks is None:
                    continue
                pred_logits = preds['mask'] if isinstance(preds, dict) else preds
                loss = _loss.loss(pred_logits, gt_masks)
            elif _loss.head == 'keypoints':
                if gt_keypoints is None:
                    continue
                if not (isinstance(preds, dict) and 'keypoints' in preds):
                    raise ValueError("Keypoints loss requested, but model did not return 'keypoints'")
                pred_kps = preds['keypoints']
                gt_kps = gt_keypoints.view(gt_keypoints.shape[0], -1)
                if kps_valid is not None:
                    pred_kps = pred_kps * kps_valid
                    gt_kps = gt_kps * kps_valid
                loss = _loss.loss(pred_kps, gt_kps)
            elif _loss.head == 'heatmaps':
                if gt_keypoints is None:
                    continue
                if not (isinstance(preds, dict) and 'heatmaps' in preds):
                    raise ValueError("Heatmap loss requested, but model did not return 'heatmaps'")
                pred_heatmaps = preds['heatmaps']
                target_heatmaps = make_gaussian_heatmaps(
                    gt_keypoints,
                    pred_heatmaps.shape[-2],
                    pred_heatmaps.shape[-1],
                    sigma=_loss.target_sigma,
                    valid=kps_valid,
                )
                loss = _loss.loss(torch.sigmoid(pred_heatmaps), target_heatmaps)
            else:
                raise ValueError(f"Unknown loss head: {_loss.head}")
            total_loss += _loss.weight * loss
            self.log(f'{prefix}{_loss.name}_loss', loss.detach())
        self.log(f'{prefix}total_loss', total_loss.detach())
        return total_loss
