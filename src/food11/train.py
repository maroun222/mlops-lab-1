"""Fine-tune pretrained ResNet18 and explicitly track training in MLflow."""
import argparse
import random
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("processed", "mini"), default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="food11")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, lr, batch-size and threads must be positive")
    return args


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count


def main():
    # MLflow prints Unicode run links; Windows redirected output can default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    folder = "food11_processed_mini" if args.dataset == "mini" else "food11_processed"
    # Keep the lab's 128x128 size; normalize RGB channels for ImageNet weights.
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    split_data = {split: datasets.ImageFolder(ROOT / "data" / folder / split, transform)
                  for split in ("training", "validation", "evaluation")}
    class_map = split_data["training"].class_to_idx
    if len(class_map) != 11 or any(ds.class_to_idx != class_map for ds in split_data.values()):
        raise ValueError("All splits must contain the same 11 category folders")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "training",
                                num_workers=0, generator=generator if split == "training" else None)
               for split, ds in split_data.items()}

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = f"resnet18_{args.dataset}_lr{args.lr:g}_bs{args.batch_size}_seed{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"dataset": args.dataset, "epochs": args.epochs, "lr": args.lr,
                           "batch_size": args.batch_size, "seed": args.seed,
                           "model": "resnet18", "weights": "IMAGENET1K_V1",
                           "optimizer": "Adam", "trainable_layers": "all",
                           "image_size": 128, "device": str(device), "threads": args.threads,
                           **{f"{s}_images": len(d) for s, d in split_data.items()}})
        mlflow.set_tag("lab", "lab2")
        mlflow.log_dict(class_map, "class_to_idx.json")
        print(f"RUN_ID={run.info.run_id}", flush=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 11)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_accuracy = 0.0
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = count = 0
            for images, labels in loaders["training"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * labels.size(0)
                count += labels.size(0)
            train_loss = train_loss_sum / count
            val_loss, val_accuracy = evaluate(model, loaders["validation"], criterion, device)
            best_val_accuracy = max(best_val_accuracy, val_accuracy)
            for name, value in {"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": val_accuracy}.items():
                mlflow.log_metric(name, value, step=epoch)
            print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}", flush=True)
        # Evaluation is the held-out test split, used only after training.
        test_loss, test_accuracy = evaluate(model, loaders["evaluation"], criterion, device)
        mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss,
                            "best_val_accuracy": best_val_accuracy,
                            "training_seconds": time.perf_counter() - started})
        model.cpu().eval()
        # MLflow 3 uses name for logged models; pickle preserves the nn.Module API.
        info = mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle",
            pip_requirements=[f"torch=={torch.__version__}",
                              f"torchvision=={__import__('torchvision').__version__}"],
        )
        mlflow.set_tag("logged_model_uri", info.model_uri)
        print(f"test_accuracy={test_accuracy:.4f} MODEL_URI={info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
