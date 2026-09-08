"""Prepare Food-11 images for ImageFolder without modifying the raw dataset."""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

CATEGORIES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
]
SPLITS = ("training", "evaluation", "validation")


def prepare(data_root: Path, mini_limit: int = 100) -> dict:
    if mini_limit < 1 or mini_limit > 100:
        raise ValueError("mini-limit must be between 1 and 100")
    raw = data_root / "food11_raw"
    full = data_root / "food11_processed"
    mini = data_root / "food11_processed_mini"
    # Fresh outputs prevent stale images from silently changing the dataset.
    if full.exists() or mini.exists():
        raise FileExistsError("Processed outputs already exist. Use a fresh data-root or restore the raw-only DVC version before rerunning.")
    inputs = {}
    for split in SPLITS:
        folder = raw / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"No images in {folder}")
        inputs[split] = []
        for path in files:
            label = int(path.stem.split("_", 1)[0])
            if not 0 <= label < len(CATEGORIES):
                raise ValueError(f"Invalid category: {path}")
            inputs[split].append((path, label))
    report = {}
    for split, files in inputs.items():
        counts = {name: 0 for name in CATEGORIES}
        for category in CATEGORIES:
            (full / split / category).mkdir(parents=True)
            (mini / split / category).mkdir(parents=True)
        for path, label in files:
            category = CATEGORIES[label]
            destination = full / split / category / path.name
            with Image.open(path) as source:
                resized = ImageOps.exif_transpose(source).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
                resized.save(destination)
            if counts[category] < mini_limit:
                shutil.copy2(destination, mini / split / category / path.name)
            counts[category] += 1
        report[split] = {"full": counts, "mini": {k: min(v, mini_limit) for k, v in counts.items()}}
        print(f"{split}: {sum(counts.values())} full, {sum(min(v, mini_limit) for v in counts.values())} mini", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--mini-limit", type=int, default=100)
    args = parser.parse_args()
    result = prepare(args.data_root, args.mini_limit)
    print(json.dumps(result, indent=2))
