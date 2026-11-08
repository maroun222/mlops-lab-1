"""Apply the Food-11 accuracy gate to a registered candidate model version."""

import argparse
import math
import os

from mlflow.exceptions import MlflowException

from mlflow import MlflowClient

MODEL_NAME = "food11"


def evaluation(client: MlflowClient, version) -> tuple[float, int, str | None]:
    if not version.run_id:
        raise ValueError(f"Version {version.version} has no associated MLflow run")

    data = client.get_run(version.run_id).data
    accuracy = data.metrics.get("test_accuracy")
    if accuracy is None or not 0 <= accuracy <= 1:
        raise ValueError(f"Version {version.version} must log test_accuracy between 0 and 1")

    try:
        sample_size = int(data.params.get("evaluation_images", ""))
    except ValueError as exc:
        raise ValueError(f"Version {version.version} must log evaluation_images") from exc
    if sample_size < 1:
        raise ValueError(f"Version {version.version} must have a positive evaluation_images count")
    return accuracy, sample_size, data.params.get("dataset")


def apply_gate(client: MlflowClient, version: int) -> str:
    candidate = client.get_model_version(MODEL_NAME, str(version))
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if candidate.version == champion.version:
        raise ValueError(f"Version {version} is already the champion")

    candidate_accuracy, candidate_n, candidate_dataset = evaluation(client, candidate)
    champion_accuracy, n, champion_dataset = evaluation(client, champion)
    if candidate_n != n or candidate_dataset != champion_dataset:
        raise ValueError("Candidate and champion must use the same evaluation dataset and sample count")

    margin = 2 * math.sqrt(champion_accuracy * (1 - champion_accuracy) / n)
    threshold = champion_accuracy + margin
    promoted = candidate_accuracy >= threshold
    alias = "champion" if promoted else "challenger"
    decision = "promoted" if promoted else "challenger"

    client.set_registered_model_alias(MODEL_NAME, alias, candidate.version)
    tags = {
        "promotion_decision": decision,
        "candidate_test_accuracy": candidate_accuracy,
        "champion_test_accuracy": champion_accuracy,
        "compared_champion_version": champion.version,
        "evaluation_images": n,
        "required_improvement": margin,
        "promotion_threshold": threshold,
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, str(value))

    return (
        f"Candidate v{candidate.version}: {candidate_accuracy:.4%}; "
        f"champion v{champion.version}: {champion_accuracy:.4%}; n={n}\n"
        f"Required improvement: {margin * 100:.4f} percentage points; "
        f"threshold: {threshold:.4%}\n"
        f"Decision: {decision}. @{alias} -> v{candidate.version}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    args = parser.parse_args()
    if args.version < 1:
        parser.error("--version must be positive")

    try:
        report = apply_gate(MlflowClient(tracking_uri=args.tracking_uri), args.version)
    except (MlflowException, ValueError) as exc:
        parser.exit(1, f"Promotion gate failed: {exc}\n")
    print(report)


if __name__ == "__main__":
    main()
