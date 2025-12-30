import asyncio
from pathlib import Path

from factory_balanced import Factory


async def main():
    documents_dir = Path("../../coco100")  # <!-- поменяй меня!!
    bg_dir = Path("../../doc100")

    if not documents_dir.exists():
        print(f"Не найдена директория с документами: {documents_dir}")
        return

    if not bg_dir.exists():
        print(f"Не найдена директория с фонами: {bg_dir}")
        return

    doc_files = list(documents_dir.glob("*.jpg")) + \
        list(documents_dir.glob("*.png"))
    bg_files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))

    print(f"Найдено {len(doc_files)} лдокументов")
    print(f"Найдено {len(bg_files)} фонов")

    if len(doc_files) == 0:
        print("Документов не найдено")
        return

    if len(bg_files) == 0:
        print("Фонов нет в папке")
        return

    factory = Factory(
        documents_dir=documents_dir,
        bg_dir=bg_dir,
        output_dir=Path("./perp"),  # <!-- смени меня!!
        iterations_per_page=3,
        canvas_scale=1.2,
        doc_scale_range=(0.6, 0.9),
        jitter_ratio_range=(0.08, 0.18),
        margin_px=100,
        jpeg_quality=95,
        dataset_name="perspective_demo",  # <!-- смени меня


        doc_aug_probability=1.0,
        use_fast_doc_aug=True,
        max_workers=4,
        batch_size=8,
    )

    print("Стартуем фабрику датасетов")

    await factory.run()

    print("Все готово")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nГенерация прервана юзером")
    except Exception as e:
        print(f"\nОшибка во время генерации {e}")
        import traceback
        traceback.print_exc()
