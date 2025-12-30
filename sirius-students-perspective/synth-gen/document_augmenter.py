"""Аугментации документов без перспективных искажений на базе Albumentations и Augraphy."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Union

import albumentations as A
# from augraphy import AugraphyPipeline
# from augraphy.augmentations import (
#     BleedThrough,
#     DirtyRollers,
#     Dithering,
#     InkBleed,
#     Letterpress,
#     LightingGradient,
#     LowInkPeriodicLines,
#     LowInkRandomLines,
#     NoiseTexturize,
# )
import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, Image.Image, np.ndarray]


class DocumentAugmenter:
    """
    Утилита для аугментации документных изображений с эффектами печати/сканирования.
    Поддерживает входные данные в виде пути к файлу, PIL.Image либо np.ndarray (RGB).
    Возвращает результат как np.ndarray либо может отдать PIL.Image/сохранить PNG с альфой.
    """

    def __init__(
        self,
        input_dir: Path = Path("documents_clean"),
        output_dir: Path = Path("documents_augmented"),
        max_images: Optional[int] = 200,
        num_augments_per_image: int = 2,
    ) -> None:
        """
        Инициализация путей и подготовка пайплайнов аугментаций.
        num_augments_per_image управляет количеством вариантов на каждое входное изображение.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_images = max_images
        self.num_augments_per_image = max(1, num_augments_per_image)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alb_transforms = self._build_alb_transforms()
        self.aug_pipeline = self._build_aug_pipeline()

    def _load_image(self, image_input: ImageInput) -> np.ndarray:
        """Загружает картинку и приводит к формату HWC, uint8, RGB, [0,255]."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Файл не найден: {path}")
            pil_img = Image.open(path).convert("RGB")
            return np.array(pil_img)
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                if image_input.dtype != np.uint8:
                    img = image_input.astype(np.float32)
                    if img.max() <= 1.0:
                        img = (img * 255).clip(0, 255)
                    return img.astype(np.uint8)
                return image_input
            raise ValueError("Ожидается массив формы (H, W, 3)")
        raise TypeError("Неподдерживаемый тип изображения. Ожидаются: str, Path, PIL.Image, np.ndarray")

    # def _build_aug_pipeline(self) -> AugraphyPipeline:
    #     """Создаёт Augraphy-пайплайн без геометрических искажений."""
    #     return AugraphyPipeline(
    #         ink_phase=[
    #             InkBleed(p=0.65),
    #             Letterpress(n_samples=(1, 2), p=0.35),
    #             LowInkRandomLines(p=0.3),
    #             LowInkPeriodicLines(p=0.25),
    #         ],
    #         paper_phase=[
    #             NoiseTexturize(
    #                 sigma_range=(3, 10),
    #                 turbulence_range=(2, 5),
    #                 texture_width_range=(100, 500),
    #                 texture_height_range=(100, 500),
    #                 p=0.4,
    #             ),
    #             BleedThrough(intensity_range=(0.05, 0.12), color_range=(64, 140), p=0.35),
    #             LightingGradient(light_position=None, min_brightness=0.7, max_brightness=1.2, p=0.4),
    #         ],
    #         post_phase=[
    #             DirtyRollers(p=0.35),
    #             Dithering(p=0.25),
    #         ],
    #         log=False,
    #     )

    def _build_alb_transforms(self) -> list[A.Compose]:
        """Два варианта цветовых/шумовых аугментаций для большего разнообразия."""
        mild = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.28, p=0.8),
                A.ColorJitter(brightness=0.18, contrast=0.22, saturation=0.18, hue=0.03, p=0.6),
                A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.25),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=14, val_shift_limit=8, p=0.3),
                A.OneOf(
                    [
                        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.6, 1.0)),
                        A.CLAHE(clip_limit=2.0),
                        A.Equalize(mode="pil"),
                    ],
                    p=0.45,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(5.0, 18.0), per_channel=False),
                        A.MultiplicativeNoise(multiplier=(0.94, 1.06), per_channel=False),
                        A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5),
                    ],
                    p=0.25,
                ),
                A.ImageCompression(quality_lower=60, quality_upper=92, p=0.4),
                A.ToGray(p=0.08),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3),
                        A.Blur(blur_limit=3),
                        A.MedianBlur(blur_limit=3),
                    ],
                    p=0.25,
                ),
            ]
        )

        heavier = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.35, p=0.9),
                A.ColorJitter(brightness=0.25, contrast=0.28, saturation=0.25, hue=0.05, p=0.7),
                A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.35),
                A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=0.4),
                A.OneOf(
                    [
                        A.Sharpen(alpha=(0.1, 0.25), lightness=(0.5, 1.1)),
                        A.UnsharpMask(p=1.0),
                        A.Emboss(alpha=(0.1, 0.2), strength=(0.1, 0.3)),
                    ],
                    p=0.55,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 28.0), per_channel=False),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.12), per_channel=False),
                        A.CoarseDropout(max_holes=8, max_height=10, max_width=10, fill_value=0, p=0.5),
                    ],
                    p=0.38,
                ),
                A.ImageCompression(quality_lower=50, quality_upper=88, p=0.5),
                A.ToGray(p=0.12),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.Blur(blur_limit=4),
                        A.MedianBlur(blur_limit=5),
                    ],
                    p=0.35,
                ),
            ]
        )

        return [mild, heavier]

    def _rgb_to_rgba(self, img_rgb: np.ndarray) -> np.ndarray:
        """Добавляет альфа-канал с полной непрозрачностью."""
        rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = 255
        return rgba

    def _apply_transforms(self, image_rgb: np.ndarray) -> np.ndarray:
        """Запускает Augraphy, затем Albumentations и возвращает готовый RGB."""
        aug_result = self.aug_pipeline.augment(image_rgb.copy())
        if isinstance(aug_result, dict):
            aug_img = aug_result.get("image", aug_result.get("output", aug_result))
        elif isinstance(aug_result, (list, tuple)):
            aug_img = aug_result[0]
        else:
            aug_img = aug_result
        if isinstance(aug_img, (list, tuple)):
            aug_img = aug_img[0]

        alb_transform = random.choice(self.alb_transforms)
        return alb_transform(image=aug_img)["image"]

    def _save_rgba(self, rgb_image: np.ndarray, output_path: Path) -> Path:
        """Сохраняет RGB массив как PNG с альфой."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba = self._rgb_to_rgba(rgb_image)
        cv2.imwrite(str(output_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        return output_path

    def apply(self, image_input: ImageInput) -> np.ndarray:
        """
        Применяет комбинированные аугментации к одному изображению.
        Возвращает np.ndarray в формате HWC, uint8, RGB.
        """
        img_rgb = self._load_image(image_input)
        return self._apply_transforms(img_rgb)

    def apply_and_return_pil(self, image_input: ImageInput) -> Image.Image:
        """Возвращает PIL.Image после аугментаций."""
        augmented = self.apply(image_input)
        return Image.fromarray(augmented)

    def apply_and_save(self, image_input: ImageInput, output_path: Union[str, Path]) -> Path:
        """Применяет аугментации и сохраняет результат в PNG с альфа-каналом."""
        augmented = self.apply(image_input)
        return self._save_rgba(augmented, Path(output_path))

    def augment_all(self) -> dict:
        """
        Обрабатывает все JPEG в input_dir, создаёт num_augments_per_image вариантов и сохраняет PNG.
        Возвращает статистику с количеством успешных аугментаций и списком ошибок.
        """
        aug_count = 0
        errors: List[tuple[str, str]] = []

        for idx, img_path in enumerate(sorted(self.input_dir.glob("*.jpg"))):
            if self.max_images is not None and idx >= self.max_images:
                break
            try:
                img_rgb = self._load_image(img_path)
            except Exception as exc:  # noqa: BLE001
                errors.append((str(img_path), f"load: {exc}"))
                continue

            for aug_idx in range(1, self.num_augments_per_image + 1):
                try:
                    augmented = self.apply(img_rgb)
                    out_path = self.output_dir / f"{img_path.stem}_aug{aug_idx}.png"
                    self._save_rgba(augmented, out_path)
                    aug_count += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append((str(img_path), f"aug{aug_idx}: {exc}"))

        return {"aug": aug_count, "errors": errors, "output_dir": str(self.output_dir)}
