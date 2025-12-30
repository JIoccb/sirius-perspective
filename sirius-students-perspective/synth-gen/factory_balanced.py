import asyncio
import csv
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm.asyncio import tqdm_asyncio
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from document_augmenter import DocumentAugmenter
from perspective_warper import PerspectiveWarper, PerspectiveWarperConfig
from result_saver import build_records, save_composite
import aiofiles


class FinalRender:
    def __init__(self):
        self.final_aug = A.Compose([A.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.02, p=0.5),
                                    A.RGBShift(
                                        r_shift_limit=6, g_shift_limit=6, b_shift_limit=6, p=0.5),
                                    A.HueSaturationValue(
            hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=6, p=0.5),
        ])


class DocumentAug:
    def __init__(self):
        self.alb_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.25, p=0.7),
            A.ColorJitter(brightness=0.15, contrast=0.2,
                          saturation=0.15, hue=0.02, p=0.5),
            A.RGBShift(r_shift_limit=6, g_shift_limit=6,
                       b_shift_limit=6, p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=6, p=0.3),
            # A.Rotate(limit=(-50, 50), p=.8),
            A.OneOf([
                A.Sharpen(alpha=(0.05, 0.1), lightness=(0.7, 1.0)),
                A.CLAHE(clip_limit=2.0),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(3.0, 12.0), per_channel=False),
                A.MultiplicativeNoise(multiplier=(
                    0.96, 1.04), per_channel=False),
            ], p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=2),
                A.Blur(blur_limit=2),
            ], p=0.15),
        ])

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.alb_transform(image=image_rgb)["image"]


class SoftShadow(ImageOnlyTransform):
    """
    Добавляет тень на документ
    """

    def __init__(
        self,
        shadow_intensity=(0.2, 0.7),
        shadow_size=(0.3, 0.8),       
        blur_sigma=(10, 30),
        direction="random",           # "left", "right", "top", "bottom", "random"
        position="random",            # "edge", "center", "random"
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.shadow_intensity = shadow_intensity
        self.shadow_size = shadow_size
        self.blur_sigma = blur_sigma
        self.direction = direction
        self.position = position

    def apply(self, img, **params):
        h, w = img.shape[:2]
        shadow = np.ones((h, w), dtype=np.float32)

        if self.direction == "random":
            direction = np.random.choice(["left", "right", "top", "bottom"])
        else:
            direction = self.direction

        if direction in ("left", "right"):
            size = int(w * np.random.uniform(*self.shadow_size))
            grad = np.linspace(1.0, np.random.uniform(
                *self.shadow_intensity), size)
            if direction == "right":
                grad = grad[::-1]
            full_grad = np.ones(w)
            if direction == "left":
                full_grad[:size] = grad
            else:
                full_grad[-size:] = grad
            shadow = np.tile(full_grad, (h, 1))

        else:
            size = int(h * np.random.uniform(*self.shadow_size))
            grad = np.linspace(1.0, np.random.uniform(
                *self.shadow_intensity), size)
            if direction == "bottom":
                grad = grad[::-1]
            full_grad = np.ones(h)
            if direction == "top":
                full_grad[:size] = grad
            else:
                full_grad[-size:] = grad
            shadow = np.tile(full_grad[:, np.newaxis], (1, w))

        sigma = np.random.uniform(*self.blur_sigma)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=sigma, sigmaY=sigma)

        img_float = img.astype(np.float32) / 255.0
        img_float *= shadow[..., np.newaxis]
        img = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return img

    def get_transform_init_args_names(self):
        return (
            "shadow_intensity",
            "shadow_size",
            "blur_sigma",
            "direction",
            "position"
        )


BACKGROUND_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DOCUMENT_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BackgroundProvider = Callable[[str], Optional[np.ndarray]]


class TaskConfig:
    def __init__(
        self,
        canvas_scale: float,
        doc_scale_range: Tuple[float, float],
        jitter_ratio_range: Tuple[float, float],
        margin_px: int,
        jpeg_quality: int,
        dataset_name: str,
    ) -> None:
        self.canvas_scale = canvas_scale
        self.doc_scale_range = doc_scale_range
        self.jitter_ratio_range = jitter_ratio_range
        self.margin_px = margin_px
        self.jpeg_quality = jpeg_quality
        self.dataset_name = dataset_name


class GenerationTask:
    def __init__(
        self,
        document_path: str,
        bg_path: str,
        filename: str,
        img_path: str,
        seed: int,
        config: TaskConfig,
    ) -> None:
        self.document_path = document_path
        self.bg_path = bg_path
        self.filename = filename
        self.img_path = img_path
        self.seed = seed
        self.config = config


class Factory:
    """Класс для наложения аугментаций v3"""

    def __init__(
        self,
        documents_dir: Path | str = Path("./documents_clean"),
        bg_dir: Path | str = Path("./COCO"),
        output_dir: Path | str = Path("./demo_perspective_datasetv2"),
        iterations_per_page: int = 1,
        canvas_scale: float = 1.2,
        doc_scale_range: Tuple[float, float] = (0.9, 1.1),
        jitter_ratio_range: Tuple[float, float] = (0.08, 0.18),
        margin_px: int = 100,
        jpeg_quality: int = 95,
        dataset_name: str = "perspective_demov2",
        background_provider: Optional[BackgroundProvider] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 8,
        doc_aug_probability: float = 0.7,
        use_fast_doc_aug: bool = True,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.bg_dir = Path(bg_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.csv_path = self.output_dir / "annotations.csv"
        self.iterations_per_page = max(1, int(iterations_per_page))
        self.background_provider = background_provider
        self.max_workers = max_workers or min(4, 8)
        self.batch_size = batch_size
        self.doc_aug_probability = doc_aug_probability
        self.use_fast_doc_aug = use_fast_doc_aug

        self.config = TaskConfig(
            canvas_scale=canvas_scale,
            doc_scale_range=doc_scale_range,
            jitter_ratio_range=jitter_ratio_range,
            margin_px=margin_px,
            jpeg_quality=jpeg_quality,
            dataset_name=dataset_name,
        )

        area_min = 0.65
        area_max = 0.85
        desired_min = float(np.sqrt(area_min) * canvas_scale)
        desired_max = float(np.sqrt(area_max) * canvas_scale)

        ds_low = max(doc_scale_range[0], desired_min)
        ds_high = min(doc_scale_range[1], desired_max)

        if ds_low > ds_high:
            ds_low, ds_high = doc_scale_range

        self.warper_config = PerspectiveWarperConfig(
            canvas_scale=canvas_scale,
            doc_scale_range=(ds_low, ds_high),
            jitter_ratio_range=jitter_ratio_range,
            margin_px=margin_px,
            page_curl_prob=0.7,
            page_curl_amount_range=(0.4, 0.7),
            page_curl_height_range=(0.2, 0.6),
        )

        # Initialize shared objects (will be created once)
        self.warper = None
        self.doc_augmenter = None
        self.light_aug = None
        self.executor = None

    def _initialize_objects(self):
        if self.warper is None:
            print("Варпер поднят")
            self.warper = PerspectiveWarper(self.warper_config)

        if self.doc_augmenter is None:
            if self.use_fast_doc_aug:
                print("Поднят быстрый аугментер")
                self.doc_augmenter = DocumentAug()
            else:
                print("Поднята тяжёлая версия аугментера")
                self.doc_augmenter = DocumentAugmenter()

        if self.light_aug is None:
            print("Поднята ауга освещения")
            self.light_aug = A.Compose([

                A.OneOf([

                    A.RGBShift(r_shift_limit=(5, 15), g_shift_limit=(
                        2, 10), b_shift_limit=(-10, -2), p=1.0),

                    A.RGBShift(r_shift_limit=(-10, -2),
                               g_shift_limit=(-5, 0), b_shift_limit=(5, 15), p=1.0),

                    A.RGBShift(r_shift_limit=(-8, -2), g_shift_limit=(8,
                               18), b_shift_limit=(-10, -5), p=1.0),

                    A.NoOp(p=1.0)
                ], p=0.6),

                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.2, p=0.6),
                A.RandomToneCurve(scale=0.05, p=0.5),
                A.GaussNoise(var_limit=(3.0, 15.0), p=0.2),


                SoftShadow(
                    shadow_intensity=(0.2, 0.5),
                    shadow_size=(0.2, 0.6),
                    blur_sigma=(10, 25),
                    direction="random",
                    p=0.25
                ),
            ], p=0.85)

        if self.executor is None:
            print(f"Поднимаем {self.max_workers} воркеров...")
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _collect_documents(self) -> List[Path]:
        documents_path = self.documents_dir
        if not documents_path.exists():
            raise SystemExit(f"Каталог документов {documents_path} не найден")

        image_files: List[Path] = []
        for file_path in sorted(documents_path.glob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in DOCUMENT_EXTS:
                continue
            image_files.append(file_path)

        if not image_files:
            raise SystemExit(
                f"В директории {documents_path} нет валидных изображений с расширениями {sorted(DOCUMENT_EXTS)}"
            )
        return image_files

    def _collect_backgrounds(self) -> List[Path]:
        bg_files = sorted(
            f
            for f in self.bg_dir.glob("*")
            if f.is_file() and f.suffix.lower() in BACKGROUND_EXTS
        )
        if not bg_files:
            raise SystemExit(f"В директории {self.bg_dir} фонов нет")
        return bg_files

    def _build_tasks(self) -> List[GenerationTask]:
        documents = self._collect_documents()
        bg_files = self._collect_backgrounds()

        total_tasks = len(documents) * self.iterations_per_page
        rng = np.random.default_rng()
        seeds = rng.integers(0, np.iinfo(np.uint32).max,
                             size=total_tasks, dtype=np.uint32)

        tasks: List[GenerationTask] = []
        seed_iter = iter(seeds)
        image_index = 0
        bg_index = 0
        bg_files_array = np.array(bg_files)

        for document_path in documents:
            for _ in range(self.iterations_per_page):
                if bg_index < len(bg_files):
                    bg_path = bg_files[bg_index]
                    bg_index += 1
                else:
                    bg_path = bg_files_array[rng.integers(0, len(bg_files))]

                filename = f"{image_index:04d}.jpg"
                img_path = str(self.images_dir / filename)
                seed = int(next(seed_iter))
                tasks.append(
                    GenerationTask(
                        document_path=str(document_path),
                        bg_path=str(bg_path),
                        filename=filename,
                        img_path=img_path,
                        seed=seed,
                        config=self.config,
                    )
                )
                image_index += 1
        return tasks

    async def _ensure_output_dirs(self) -> None:
        """Асинх проверяем что директории существуют"""
        print("Проверяем существование директорий")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)

    def _process_single_task(self, task: GenerationTask) -> Optional[dict]:
        try:
            document_np = cv2.imread(task.document_path)
            if document_np is None:
                print(f"Не удалось загрузить документ: {task.document_path}")
                return None

            if self.background_provider:
                background_np = self.background_provider(task.bg_path)
            else:
                background_np = cv2.imread(task.bg_path)

            if background_np is None:
                print(f"Не удалось загрузить бэкграунд: {task.bg_path}")
                return None

            task_rng = np.random.default_rng(task.seed)
            if task_rng.random() < self.doc_aug_probability:
                doc_rgb = cv2.cvtColor(document_np, cv2.COLOR_BGR2RGB)
                doc_aug_rgb = self.doc_augmenter.apply(doc_rgb)
                document_np = cv2.cvtColor(doc_aug_rgb, cv2.COLOR_RGB2BGR)

            rng = np.random.default_rng(int(task.seed))
            (
                composite,
                keypoints_before,
                keypoints_after,
                canvas_w,
                canvas_h,
                warped_mask,
            ) = self.warper.apply(document_np, background_np, rng=rng)

            try:
                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                aug_rng = np.random.default_rng(task.seed + 42)

                if self.light_aug is not None:
                    augmented = self.light_aug(
                        image=composite_rgb,
                        seed=int(aug_rng.integers(0, 2**32 - 1))
                    )
                    composite_aug_rgb = augmented['image']
                else:
                    composite_aug_rgb = composite_rgb

                final_render = FinalRender()
                composite_final_rgb = final_render.final_aug(
                    image=composite_aug_rgb)['image']

                composite_final = cv2.cvtColor(
                    composite_final_rgb, cv2.COLOR_RGB2BGR)

            except Exception as e:
                print(
                    f"Ошибка при накидывании аугмнентаций {task.filename}: {str(e)}")
                composite_final = composite

            return {
                'task': task,
                'composite_aug': composite_final,
                'warped_mask': warped_mask,
                'keypoints_before': keypoints_before,
                'keypoints_after': keypoints_after,
                'canvas_w': canvas_w,
                'canvas_h': canvas_h,
            }

        except Exception as e:
            print(f"Error processing task {task.filename}: {str(e)}")
            return None

    async def _save_results(self, result_data) -> Optional[Tuple[List, dict]]:
        """Асинхронно всё сохраняем"""
        try:
            task = result_data['task']
            composite_aug = result_data['composite_aug']
            warped_mask = result_data['warped_mask']
            keypoints_before = result_data['keypoints_before']
            keypoints_after = result_data['keypoints_after']
            canvas_w = result_data['canvas_w']
            canvas_h = result_data['canvas_h']

            cv2.imwrite(
                task.img_path,
                composite_aug,
                [cv2.IMWRITE_JPEG_QUALITY, task.config.jpeg_quality]
            )

            masks_dir = self.output_dir / "masks"
            mask_path = masks_dir / (Path(task.filename).stem + ".png")
            warped_mask_u8 = (warped_mask > 0).astype(np.uint8) * 255
            cv2.imwrite(str(mask_path), warped_mask_u8)

            row, annotation = build_records(
                task.filename,
                Path(task.img_path),
                canvas_w,
                canvas_h,
                keypoints_before,
                keypoints_after,
                task.config.dataset_name,
            )

            return row, annotation

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return None

    async def _write_csv(self, rows: List) -> None:
        """Записываем csv"""
        async with aiofiles.open(self.csv_path, "w", newline="", encoding="utf-8") as f:

            await f.write("filename,filepath,width,height,coords,visibility,dataset\n")

            for row in rows:
                row_str = ",".join(str(item) for item in row) + "\n"
                await f.write(row_str)

    async def generate(self) -> None:
        """Главный метод"""
        await self._ensure_output_dirs()
        self._initialize_objects()

        tasks = self._build_tasks()
        rows = []
        first_annotation = None

        batches = [tasks[i:i + self.batch_size]
                   for i in range(0, len(tasks), self.batch_size)]

        for batch_idx, batch in enumerate(tqdm_asyncio(batches, desc="Обрабатываем батчи")):

            loop = asyncio.get_event_loop()
            batch_futures = [
                loop.run_in_executor(
                    self.executor, self._process_single_task, task)
                for task in batch
            ]

            batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)

            save_futures = []
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    save_futures.append(self._save_results(result))

            if save_futures:
                saved_results = await asyncio.gather(*save_futures, return_exceptions=True)

                for saved_result in saved_results:
                    if saved_result is not None and not isinstance(saved_result, Exception):
                        row, annotation = saved_result
                        rows.append(row)
                        if first_annotation is None:
                            first_annotation = annotation

        if not rows:
            raise SystemExit("Не удалось сгенерировать ни одного изображения")

        print("Записываем CSV")
        rows.sort(key=lambda r: r[0])
        await self._write_csv(rows)

        print(f"Сохранено изображений: {len(rows)}")
        print(f"Папка с изображениями: {self.images_dir.resolve()}")
        print(f"CSV с аннотациями: {self.csv_path.resolve()}")

        if first_annotation:
            print("\nПример аннотации:")
            print(json.dumps(first_annotation, indent=2, ensure_ascii=False))

    async def run(self) -> None:
        try:
            await self.generate()
        finally:
            if self.executor:
                print("Стопаем экзекьютор")
                self.executor.shutdown(wait=False)


class PerspectiveFactory:
    def __init__(self, **kwargs):
        self.balanced_factory = Factory(**kwargs)

    def generate(self):
        asyncio.run(self.balanced_factory.generate())

    def run(self):
        asyncio.run(self.balanced_factory.run())


if __name__ == "__main__":
    asyncio.run(Factory(
        bg_dir=Path("./bg"),
        iterations_per_page=2
    ).run())
