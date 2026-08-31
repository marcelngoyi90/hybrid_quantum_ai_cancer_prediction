"""Recreate missing downstream states without accessing locked-test data.

This recovery is an explicitly documented exception: the development runner
saved aggregate results but not model state files.  It uses only the original
development feature handoff and refuses to save a state unless its selected
epoch and ROC-AUC reproduce the recorded development result.
"""

import hashlib
import json
from pathlib import Path

import torch

from data_loader import load_feature_archive
from models import ClassicalNeuralModel, HybridQuantumModel
from preprocessing import transform_features_to_angles
from training import set_random_seed, train_binary_model


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiment_config.json"
RESULTS_ROOT = ROOT / "results" / "development"
STATE_ROOT = ROOT / "private_checkpoints"
AUDIT_PATH = ROOT / "results" / "frozen_state_recovery_audit.json"

AUC_TOLERANCE = 1e-6
SEEDS = [42, 123, 2026]
TASKS = {
    "camelyon": {
        "train_file": "image_train_8d.npz",
        "selection_file": "image_ood_val_8d.npz",
    },
    "seer": {
        "train_file": "tabular_train_8d.npz",
        "selection_file": "tabular_temporal_val_8d.npz",
    },
}
MODEL_FACTORIES = {
    "classical_neural": ClassicalNeuralModel,
    "hybrid_quantum": HybridQuantumModel,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


with CONFIG_PATH.open("r", encoding="utf-8") as stream:
    config = json.load(stream)

training_config = config["training"]
STATE_ROOT.mkdir(parents=True, exist_ok=True)

audit = {
    "phase": "frozen_downstream_state_recovery",
    "reason": (
        "Development metrics were saved, but downstream state files were "
        "not persisted before the locked handoff arrived."
    ),
    "authorization": (
        "Upstream owner instructed the contributor to complete the locked "
        "test with the frozen quantum architecture."
    ),
    "locked_arrays_loaded": False,
    "architecture_changed": False,
    "hyperparameters_changed": False,
    "development_config_sha256": sha256_file(CONFIG_PATH),
    "auc_tolerance": AUC_TOLERANCE,
    "states": [],
}

for task_name, task_files in TASKS.items():
    for seed in SEEDS:
        recorded_path = RESULTS_ROOT / (
            f"{task_name}_seed{seed}_classical_vs_quantum.json"
        )
        with recorded_path.open("r", encoding="utf-8") as stream:
            recorded = json.load(stream)

        train_features, train_labels = load_feature_archive(
            seed, task_files["train_file"]
        )
        validation_features, validation_labels = load_feature_archive(
            seed, task_files["selection_file"]
        )
        train_features = transform_features_to_angles(train_features)
        validation_features = transform_features_to_angles(validation_features)

        for model_name, model_factory in MODEL_FACTORIES.items():
            print("=" * 72)
            print(f"Recovering {task_name} seed {seed} {model_name}")

            set_random_seed(seed)
            model = model_factory()
            recovered = train_binary_model(
                model=model,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                seed=seed,
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                maximum_epochs=training_config["maximum_epochs"],
                patience=training_config["early_stopping_patience"],
            )

            expected = recorded["models"][model_name]
            expected_epoch = int(expected["best_epoch"])
            expected_auc = float(expected["best_validation_roc_auc"])
            recovered_epoch = int(recovered["best_epoch"])
            recovered_auc = float(recovered["best_validation_roc_auc"])
            auc_difference = abs(recovered_auc - expected_auc)

            if recovered_epoch != expected_epoch:
                raise RuntimeError(
                    f"Best epoch mismatch for {task_name} seed {seed} "
                    f"{model_name}: expected {expected_epoch}, got "
                    f"{recovered_epoch}. No checkpoint was accepted."
                )
            if auc_difference > AUC_TOLERANCE:
                raise RuntimeError(
                    f"ROC-AUC mismatch for {task_name} seed {seed} "
                    f"{model_name}: expected {expected_auc:.12f}, got "
                    f"{recovered_auc:.12f}. No checkpoint was accepted."
                )

            state_path = STATE_ROOT / (
                f"{task_name}_seed{seed}_{model_name}.pt"
            )
            torch.save(
                {
                    "task": task_name,
                    "seed": seed,
                    "model_name": model_name,
                    "best_epoch": recovered_epoch,
                    "development_roc_auc": recovered_auc,
                    "development_config_sha256": audit[
                        "development_config_sha256"
                    ],
                    "model_state_dict": recovered["model"].state_dict(),
                },
                state_path,
            )
            state_hash = sha256_file(state_path)

            audit["states"].append(
                {
                    "task": task_name,
                    "seed": seed,
                    "model_name": model_name,
                    "best_epoch": recovered_epoch,
                    "recorded_development_roc_auc": expected_auc,
                    "recovered_development_roc_auc": recovered_auc,
                    "absolute_auc_difference": auc_difference,
                    "reproduction_passed": True,
                    "state_filename": state_path.name,
                    "state_sha256": state_hash,
                }
            )
            print(f"Accepted state: {state_path.name}")
            print(f"SHA-256: {state_hash}")

AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
with AUDIT_PATH.open("w", encoding="utf-8") as stream:
    json.dump(audit, stream, indent=2)

print("=" * 72)
print(f"Recovered and verified {len(audit['states'])} frozen states.")
print(f"Public audit saved to: {AUDIT_PATH}")
print("No locked-test array was loaded during state recovery.")
