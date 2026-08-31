"""One-time evaluation of verified frozen states on locked-test features.

This file intentionally contains no optimizer, loss, backward pass, fitting,
or training call. It validates handoff hashes, loads verified frozen states,
and evaluates each task/model/seed exactly once at threshold 0.5.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from metrics import calculate_binary_metrics, print_binary_metrics
from models import ClassicalNeuralModel, HybridQuantumModel
from preprocessing import transform_features_to_angles
from training import predict_probabilities


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
HANDOFF_ROOT = WORKSPACE_ROOT / "joint_v2_final_locked_handoff"
FEATURE_ROOT = HANDOFF_ROOT / "artifacts" / "features" / "joint_v2"
AUDIT_ROOT = (
    HANDOFF_ROOT
    / "artifacts"
    / "metrics"
    / "joint_v2"
    / "locked_feature_extraction"
)
STATE_ROOT = ROOT / "private_checkpoints"
RECOVERY_AUDIT_PATH = ROOT / "results" / "frozen_state_recovery_audit.json"
OUTPUT_ROOT = ROOT / "results" / "locked_test"

SEEDS = [42, 123, 2026]
TASKS = {
    "camelyon": {
        "upstream_name": "image",
        "filename": "image_ood_test_8d.npz",
        "expected_rows": 85054,
    },
    "seer": {
        "upstream_name": "tabular",
        "filename": "tabular_temporal_test_8d.npz",
        "expected_rows": 111496,
    },
}
MODEL_FACTORIES = {
    "classical_neural": ClassicalNeuralModel,
    "hybrid_quantum": HybridQuantumModel,
}
METRIC_NAMES = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_and_validate_locked_archive(task_name, seed):
    task = TASKS[task_name]
    archive_path = FEATURE_ROOT / f"seed{seed}" / task["filename"]
    audit_path = AUDIT_ROOT / (
        f"seed{seed}_{task['upstream_name']}_locked_features_audit.json"
    )
    with audit_path.open("r", encoding="utf-8") as stream:
        upstream_audit = json.load(stream)
    expected = upstream_audit["outputs"]["reduced_8d"]

    actual_file_hash = sha256_file(archive_path)
    if actual_file_hash != expected["file_sha256"]:
        raise RuntimeError(f"Locked archive hash mismatch: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "labels"}:
            raise RuntimeError(f"Unexpected arrays in {archive_path}")
        features = archive["features"]
        labels = archive["labels"]

    if features.shape != (task["expected_rows"], 8):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")
    if labels.shape != (task["expected_rows"],):
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")
    if features.dtype != np.float32 or labels.dtype != np.uint8:
        raise RuntimeError("Locked feature or label dtype mismatch")
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Invalid value in locked archive")
    if sha256_array(features) != expected["features_sha256"]:
        raise RuntimeError("Locked feature-array hash mismatch")
    if sha256_array(labels) != expected["labels_sha256"]:
        raise RuntimeError("Locked label-array hash mismatch")

    return features, labels, actual_file_hash


with RECOVERY_AUDIT_PATH.open("r", encoding="utf-8") as stream:
    recovery_audit = json.load(stream)

if recovery_audit["locked_arrays_loaded"] is not False:
    raise RuntimeError("Recovery audit does not prove locked-data separation")
if len(recovery_audit["states"]) != 12:
    raise RuntimeError("Exactly 12 verified frozen states are required")

verified_states = {
    (item["task"], item["seed"], item["model_name"]): item
    for item in recovery_audit["states"]
    if item["reproduction_passed"] is True
}
if len(verified_states) != 12:
    raise RuntimeError("Every frozen state must pass development reproduction")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
all_results = []

for task_name, task in TASKS.items():
    for seed in SEEDS:
        print("=" * 72)
        print(f"Loading locked {task_name} features for seed {seed}")
        raw_features, labels, archive_hash = load_and_validate_locked_archive(
            task_name, seed
        )
        features = transform_features_to_angles(raw_features)

        for model_name, model_factory in MODEL_FACTORIES.items():
            state_audit = verified_states[(task_name, seed, model_name)]
            state_path = STATE_ROOT / state_audit["state_filename"]
            if sha256_file(state_path) != state_audit["state_sha256"]:
                raise RuntimeError(f"Frozen state hash mismatch: {state_path}")

            model = model_factory()
            checkpoint = torch.load(
                state_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            probabilities = predict_probabilities(
                model=model,
                features=features,
                batch_size=4096,
            )
            metrics = calculate_binary_metrics(
                labels=labels,
                probabilities=probabilities,
                threshold=0.5,
            )
            print_binary_metrics(
                f"LOCKED {task_name} seed {seed} {model_name}", metrics
            )

            all_results.append(
                {
                    "phase": "final_locked_test",
                    "task": task_name,
                    "seed": seed,
                    "model_name": model_name,
                    "threshold": 0.5,
                    "locked_archive_sha256": archive_hash,
                    "frozen_state_sha256": state_audit["state_sha256"],
                    "metrics": metrics,
                }
            )

summary = {}
for task_name in TASKS:
    summary[task_name] = {}
    for model_name in MODEL_FACTORIES:
        rows = [
            row
            for row in all_results
            if row["task"] == task_name and row["model_name"] == model_name
        ]
        summary[task_name][model_name] = {}
        for metric_name in METRIC_NAMES:
            values = np.array(
                [row["metrics"][metric_name] for row in rows], dtype=float
            )
            summary[task_name][model_name][metric_name] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": float(values.std(ddof=1)),
                "values": values.tolist(),
            }

    summary[task_name]["quantum_minus_classical"] = {}
    for metric_name in METRIC_NAMES:
        classical = np.array(
            [
                row["metrics"][metric_name]
                for row in all_results
                if row["task"] == task_name
                and row["model_name"] == "classical_neural"
            ],
            dtype=float,
        )
        quantum = np.array(
            [
                row["metrics"][metric_name]
                for row in all_results
                if row["task"] == task_name
                and row["model_name"] == "hybrid_quantum"
            ],
            dtype=float,
        )
        differences = quantum - classical
        summary[task_name]["quantum_minus_classical"][metric_name] = {
            "mean": float(differences.mean()),
            "sample_standard_deviation": float(differences.std(ddof=1)),
            "values": differences.tolist(),
        }

payload = {
    "phase": "final_locked_test",
    "one_time_evaluation": True,
    "training_performed": False,
    "threshold_tuned": False,
    "architecture_changed": False,
    "individual_results": all_results,
    "three_seed_summary": summary,
}
output_path = OUTPUT_ROOT / "final_locked_test_results.json"
with output_path.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, default=float)

print("=" * 72)
for task_name in TASKS:
    print(task_name.upper())
    for model_name in MODEL_FACTORIES:
        result = summary[task_name][model_name]["roc_auc"]
        print(
            f"{model_name} ROC-AUC: {result['mean']:.6f} +/- "
            f"{result['sample_standard_deviation']:.6f}"
        )
    difference = summary[task_name]["quantum_minus_classical"]["roc_auc"]
    print(
        f"quantum_minus_classical ROC-AUC: {difference['mean']:.6f} +/- "
        f"{difference['sample_standard_deviation']:.6f}"
    )
print(f"Final locked results saved to: {output_path}")
print("No training, fitting, tuning, or threshold selection was performed.")
