import argparse
import asyncio
import math
from pathlib import Path

from factory_balanced import Factory


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a small local synthetic document perspective dataset.",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=repo_root / "raw_dataset" / "WarpDoc" / "WarpDoc" / "digital" / "perspective",
        help="Directory with clean document renders.",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=Path,
        default=repo_root / "raw_dataset" / "val2014" / "val2014",
        help="Directory with background images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "synthetic_dataset",
        help="Output dataset directory.",
    )
    parser.add_argument("--samples", type=int, default=30, help="Number of images to generate.")
    parser.add_argument(
        "--iterations-per-page",
        type=int,
        default=None,
        help="Augmented variants per source document. Defaults to enough variants to reach --samples.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for task sampling.")
    parser.add_argument("--workers", type=int, default=4, help="Thread workers.")
    parser.add_argument("--batch-size", type=int, default=8, help="Async batch size.")
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="Resize saved images and masks to WIDTH HEIGHT. Use 0 0 to keep original canvas size.",
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()

    if not args.documents_dir.exists():
        raise SystemExit(f"Documents directory does not exist: {args.documents_dir}")
    if not args.backgrounds_dir.exists():
        raise SystemExit(f"Backgrounds directory does not exist: {args.backgrounds_dir}")

    doc_count = len([
        path for path in args.documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    if doc_count == 0:
        raise SystemExit(f"No document images found in: {args.documents_dir}")

    iterations_per_page = args.iterations_per_page
    if iterations_per_page is None:
        iterations_per_page = max(1, math.ceil(args.samples / doc_count))

    output_size = tuple(args.output_size)
    if output_size == (0, 0):
        output_size = None

    factory = Factory(
        documents_dir=args.documents_dir,
        bg_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        iterations_per_page=iterations_per_page,
        max_samples=args.samples,
        seed=args.seed,
        output_size=output_size,
        canvas_scale=1.2,
        doc_scale_range=(0.6, 0.9),
        jitter_ratio_range=(0.08, 0.18),
        margin_px=100,
        jpeg_quality=95,
        dataset_name="synthetic_local",
        doc_aug_probability=1.0,
        use_fast_doc_aug=True,
        max_workers=args.workers,
        batch_size=args.batch_size,
    )
    await factory.run()


if __name__ == "__main__":
    asyncio.run(main())
