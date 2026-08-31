# Joint Model v2: Classical vs Hybrid Quantum Classifiers

## Purpose

This directory contains the downstream classical-versus-quantum comparison for the eight-dimensional representations produced by joint model v2.

The upstream model used hard-parameter-sharing multi-task learning on two unpaired datasets:

- Camelyon17-WILDS histopathology image patches.
- SEER breast-cancer registry records.

The datasets do not contain matched patients and must not be combined row by row. They remain separate prediction tasks.

## Prediction tasks

### Camelyon

The Camelyon task predicts whether a histopathology patch contains metastatic tumour tissue in its labelled central region.

Development model selection uses the out-of-distribution validation split.

### SEER

The SEER task predicts recorded regional lymph-node involvement among patients already diagnosed with breast cancer:

- Class 0: N0.
- Class 1: N+.

Development model selection uses the temporal validation split.

## Private feature handoff

For each task, split and upstream seed, the private handoff contains:

- `features`: an `N x 8` `float32` matrix.
- `labels`: an `N`-element binary `uint8` vector.

The three upstream seeds are:

- 42
- 123
- 2026

Private `.npz` archives and SEER-derived row-level information must not be committed to Git.

## Input transformation

Both downstream models receive the same transformed eight-dimensional inputs.

The deterministic transformation is:

```text
angle = 2 * arctan(feature)
```

This maps the real-valued features into approximately `[-pi, pi]`.

The transformation:

- Does not use labels.
- Does not use validation data to fit parameters.
- Does not use locked-test information.
- Is applied identically to the classical and quantum models.

## Classical neural model

The matched classical model uses:

```text
8 inputs -> 3 tanh hidden units -> 1 output logit
```

Trainable parameter count: 31.

## Hybrid quantum model

The hybrid model uses:

- Four simulated qubits.
- Two data-reuploading rounds.
- Features 1-4 encoded using Y rotations.
- One strongly entangling layer.
- Features 5-8 encoded using Y rotations.
- A second strongly entangling layer.
- Four Pauli-Z expectation values.
- A final classical `4 -> 1` output layer.

Trainable parameter count: 29.

The circuit uses PennyLane's `default.qubit` simulator and PyTorch backpropagation.

## Frozen training configuration

Both models use:

- Adam optimizer.
- Learning rate: 0.01.
- Batch size: 1024.
- Maximum epochs: 15.
- Early-stopping patience: 4.
- Weighted binary cross-entropy with logits.
- Validation ROC-AUC for checkpoint selection.
- Classification threshold: 0.5.

The same rows, labels, input transformation, loss family, optimizer, batch size, model-selection metric and threshold are used for both models.

## Development results

Results are reported as mean +/- sample standard deviation across upstream seeds 42, 123 and 2026.

### Camelyon OOD validation

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Classical neural | 0.978806 +/- 0.002870 | 0.982347 +/- 0.002362 | 0.930810 +/- 0.006956 | 0.930810 +/- 0.006956 |
| Hybrid quantum | 0.974687 +/- 0.002632 | 0.978456 +/- 0.001267 | 0.929273 +/- 0.005517 | 0.929273 +/- 0.005517 |

Mean quantum-minus-classical ROC-AUC:

```text
-0.004118 +/- 0.002229
```

### SEER temporal validation

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Classical neural | 0.794835 +/- 0.001341 | 0.654786 +/- 0.002128 | 0.709393 +/- 0.002983 | 0.715589 +/- 0.001132 |
| Hybrid quantum | 0.792162 +/- 0.001102 | 0.642896 +/- 0.004701 | 0.705896 +/- 0.009226 | 0.714058 +/- 0.000443 |

Mean quantum-minus-classical ROC-AUC:

```text
-0.002673 +/- 0.000615
```

## Development interpretation

The hybrid quantum model learned strong predictive signal on both tasks, but it did not outperform the matched classical neural model on mean development ROC-AUC.

The development evidence therefore does not demonstrate a quantum advantage.

This negative comparison is a valid scientific result and must not be hidden or changed by selecting only a favorable seed.

## Final locked-test status

After the downstream procedure was frozen and approved, the Camelyon `ood_test`
and SEER `temporal_test` representations were evaluated once. No training,
fitting, tuning, calibration, threshold selection, architecture change, seed
selection, or preprocessing change was performed using locked-test data.

Final three-seed ROC-AUC results were:

| Task | Classical neural | Hybrid quantum | Quantum minus classical |
| --- | ---: | ---: | ---: |
| Camelyon OOD test | 0.979976 +/- 0.001784 | 0.973123 +/- 0.006888 | -0.006853 +/- 0.005617 |
| SEER temporal test | 0.797122 +/- 0.001216 | 0.794584 +/- 0.000899 | -0.002537 +/- 0.000424 |

The full final report documents the state-recovery disclosure, integrity rules,
supporting metrics, and interpretation.

## Main files

- `data_loader.py`: validates and loads private feature archives.
- `preprocessing.py`: applies the shared angle transformation.
- `models.py`: defines the matched classical and hybrid quantum models.
- `training.py`: defines reproducible training, prediction and early stopping.
- `metrics.py`: calculates binary-classification metrics.
- `run_development_experiment.py`: trains one task and seed.
- `summarize_development.py`: aggregates the three-seed results.
- `experiment_config.json`: records the frozen experiment configuration.
- `results/development/`: contains privacy-safe aggregate development results.
- `recover_frozen_states.py`: recreates and verifies missing frozen downstream states without loading locked arrays.
- `run_locked_test.py`: performs evaluation-only locked inference with verified states and archives.
- `results/locked_test/`: contains the privacy-safe final result JSON and report.

## Running a development experiment

In `run_development_experiment.py`, set:

```python
TASK_NAME = "camelyon"
SEED = 42
```

Allowed task names are:

- `camelyon`
- `seer`

Allowed seeds are:

- `42`
- `123`
- `2026`

Run a development experiment with:

```text
python joint_v2/run_development_experiment.py
```

Generate the three-seed summary with:

```text
python joint_v2/summarize_development.py
```

## Research limitations

- The final results use one-time locked tests but no additional external validation cohort.
- The quantum circuit is simulated.
- Only one small quantum architecture and one parameter-matched classical architecture were evaluated.
- The two datasets are unpaired and represent different prediction targets.
- SEER features remain subject to privacy and data-use restrictions.
- The study does not establish clinical utility.
