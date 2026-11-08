"""Check the Food-11 data contract for a processed dataset folder."""

import argparse
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, SPLITS
else:
    from data import CLASSES, SPLITS


def validate(root: Path, min_per_class: int, tau: float) -> bool:
    expected = set(CLASSES)
    folder_errors = []
    image_errors = []
    count_errors = []
    duplicate_errors = []
    share_errors = []
    counts = {}
    hashes = {}
    files_checked = 0

    for split in SPLITS:
        split_path = root / split
        actual = {path.name for path in split_path.iterdir() if path.is_dir()} if split_path.is_dir() else set()
        if actual != expected:
            folder_errors.append(
                f"{split}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

        counts[split] = {}
        for category in CLASSES:
            folder = split_path / category
            images = sorted(path for path in folder.rglob("*") if path.is_file())
            counts[split][category] = len(images)
            if len(images) < min_per_class:
                count_errors.append(f"{split}/{category}: {len(images)} < {min_per_class}")

            for path in images:
                files_checked += 1
                relative_path = path.relative_to(root)
                try:
                    content = path.read_bytes()
                    digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
                    first_split, first_path = hashes.setdefault(digest, (split, relative_path))
                    if first_split != split:
                        duplicate_errors.append(f"{first_path} matches {relative_path}")
                    with Image.open(BytesIO(content)) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (128, 128):
                            image_errors.append(f"{relative_path}: mode={image.mode}, size={image.size}")
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                    image_errors.append(f"{relative_path}: {exc}")

    training_total = sum(counts["training"].values())
    validation_total = sum(counts["validation"].values())
    if not training_total or not validation_total:
        share_errors.append("training and validation must both contain images")
    else:
        for category in CLASSES:
            training_share = counts["training"][category] / training_total
            validation_share = counts["validation"][category] / validation_total
            difference = abs(training_share - validation_share)
            if difference > tau:
                share_errors.append(f"{category}: share difference {difference:.4f} > {tau:.4f}")

    checks = (
        ("Class folders", folder_errors),
        (f"Image format and dimensions ({files_checked} files)", image_errors),
        (f"Minimum images per class ({min_per_class})", count_errors),
        ("No duplicate files across splits", duplicate_errors),
        (f"Training/validation class shares (tau={tau:g})", share_errors),
    )
    for name, errors in checks:
        if errors:
            print(f"FAIL {name}: {len(errors)} issue(s); first: {errors[0]}")
        else:
            print(f"PASS {name}")
    return not any(errors for _, errors in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Processed dataset folder")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    if args.min_per_class < 1:
        parser.error("--min-per-class must be positive")
    if not 0 <= args.tau <= 1:
        parser.error("--tau must be between 0 and 1")
    return 0 if validate(args.dataset, args.min_per_class, args.tau) else 1


if __name__ == "__main__":
    raise SystemExit(main())
