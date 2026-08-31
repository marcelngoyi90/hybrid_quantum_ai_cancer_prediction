# Hybrid Quantum AI for Cancer Prediction

## Project overview

This repository contains a joint cancer-AI pipeline trained on two large, unpaired datasets:

- Camelyon17-WILDS histopathology image patches.
- SEER breast-cancer registry records.

The datasets are not merged row by row and do not contain the same patients. Instead, the model uses separate image and tabular encoders that feed a shared neural representation.

This part of the project covers dataset preparation, joint-model training, development evaluation, shared-feature extraction, dimensionality reduction and secure feature handoff.

## Prediction tasks

| Dataset | Input | Target |
|---|---|---|
| Camelyon17-WILDS | 96 × 96 lymph-node histopathology patch | Class 0: no metastatic tumour in the labelled central region; Class 1: metastatic tumour |
| SEER breast cancer | Demographic and tumour characteristics | N0: no positive regional lymph nodes; N+: at least one positive regional lymph node |

All SEER patients already have a breast-cancer diagnosis. The SEER model does not predict whether a person has breast cancer or is cancer-free.

## Dataset sizes

### Camelyon17-WILDS

| Split | Patches | Hospitals | Status |
|---|---:|---|---|
| Training | 302,436 | 0, 3, 4 | Development |
| Internal validation | 33,560 | 0, 3, 4 | Development |
| OOD validation | 34,904 | 1 | Development |
| OOD test | 85,054 | 2 | Locked |

### SEER

The raw SEER export contained 756,356 tumour records. Deterministic index-record selection produced 731,278 unique adult patients.

| Split | Patients | Diagnosis years | Status |
|---|---:|---|---|
| Training | 461,292 | 2010–2019 | Development |
| Internal validation | 51,241 | 2010–2019 | Development |
| Temporal validation | 107,249 | 2020–2021 | Development |
| Temporal test | 111,496 | 2022–2023 | Locked |

The final SEER cohort contains:

- 503,794 N0 patients.
- 227,484 N+ patients.
- 31.11% N+ prevalence.

## Joint model

The model uses hard parameter sharing across two unpaired tasks.

### Image path

```text
96 × 96 × 3 image
→ DenseNet121
→ 1024-dimensional image representation
→ 1024 → 256 → 128 image projector
```

DenseNet121 is initialised with ImageNet weights, and all layers are fine-tuned.

### Tabular path

The SEER model uses:

- 2 numeric features.
- 13 categorical features with learned embeddings.
- A 256 → 256 neural encoder.
- A 256 → 128 projector.

### Shared path

```text
128-dimensional modality representation
→ shared trunk: 128 → 64 → 32
→ separate 32 → 1 task heads
```

Each training step processes one image batch and one SEER batch. The two losses receive equal weight and update the shared trunk together.

## Model selection

The best checkpoint is selected using:

```text
0.5 × Camelyon OOD-validation ROC-AUC
+ 0.5 × SEER temporal-validation ROC-AUC
```

The model was trained independently with seeds:

```text
42, 123, 2026
```

## Development results

Results are mean ± sample standard deviation across the three seeds.

| Task and split | ROC-AUC | Accuracy |
|---|---:|---:|
| Camelyon OOD validation | 0.9789 ± 0.0029 | 92.84% ± 1.15% |
| SEER temporal validation | 0.7946 ± 0.0016 | 75.99% ± 0.13% |

At the default SEER threshold of 0.5:

- Sensitivity: 43.96%.
- Specificity: 90.30%.
- Balanced accuracy: 67.13%.

Thresholds selected using internal validation produced:

- Mean threshold: 0.311.
- Sensitivity: 74.63%.
- Specificity: 68.31%.
- Balanced accuracy: 71.47%.
- F1: 60.78%.

This threshold analysis changes the operating trade-off but does not retrain the model or change ROC-AUC.

## Shared-feature extraction

The selected checkpoint for each seed extracts a 32-dimensional shared representation.

Extraction is performed with:

- Evaluation mode enabled.
- Model parameters frozen.
- No additional training.
- Original row order preserved.
- No patient identifiers loaded or stored.
- Locked tests excluded.

## Eight-dimensional representation

A separate StandardScaler and PCA pipeline is fitted for each task and seed using training representations only.

```text
32-dimensional shared representation
→ training-only standardisation
→ training-only PCA
→ 8-dimensional representation
```

All six PCA pipelines retained more than 99.91% of total variance.

A controlled linear-probe comparison measured predictive retention:

| Task and development split | 32D ROC-AUC | 8D ROC-AUC | Retention |
|---|---:|---:|---:|
| Camelyon OOD validation | 0.980280 | 0.978598 | 99.83% |
| SEER temporal validation | 0.794814 | 0.794643 | 99.98% |

The private feature archives are not committed to GitHub.

## Final locked-test evaluation

After the upstream checkpoints, training-fitted PCA pipelines and downstream classifier procedure were frozen, Camelyon `ood_test` and SEER `temporal_test` were evaluated once.

Results are mean ± sample standard deviation across seeds 42, 123 and 2026.

| Task | Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|
| Camelyon OOD test | Classical neural | 0.979976 ± 0.001784 | 0.982969 ± 0.001511 | 0.918440 ± 0.005571 | 0.918440 ± 0.005571 | 0.913276 ± 0.006783 |
| Camelyon OOD test | Hybrid quantum | 0.973123 ± 0.006888 | 0.977738 ± 0.005096 | 0.919075 ± 0.010576 | 0.919075 ± 0.010576 | 0.914269 ± 0.013010 |
| SEER temporal test | Classical neural | 0.797122 ± 0.001216 | 0.653177 ± 0.001943 | 0.712052 ± 0.002791 | 0.718665 ± 0.000696 | 0.605214 ± 0.000834 |
| SEER temporal test | Hybrid quantum | 0.794584 ± 0.000899 | 0.642101 ± 0.003616 | 0.707873 ± 0.010166 | 0.716654 ± 0.001185 | 0.602893 ± 0.001385 |

The hybrid model did not outperform the parameter-matched classical model on the predeclared primary ROC-AUC metric:

- Camelyon quantum-minus-classical ROC-AUC: −0.006853 ± 0.005617.
- SEER quantum-minus-classical ROC-AUC: −0.002537 ± 0.000424.

### Downstream state-recovery disclosure

The original downstream development runner saved metrics but did not preserve the selected model-state files. Before any locked array was loaded, all 12 downstream states were recreated using only the frozen development data, seeds, architectures, training settings and model-selection rules.

A recovered state was accepted only when its selected epoch and development ROC-AUC reproduced the recorded result within an absolute tolerance of `1e-6`. All 12 states passed and were frozen by SHA-256 hash before locked-test evaluation.

This recovery was a documented procedural exception. It did not use locked-test data for training, fitting, tuning, model selection or threshold selection.

## Repository structure

```text
configs/       Frozen experiment and preprocessing configurations
notebooks/     Reproducible dataset and model documentation
scripts/       Dataset preparation, training and evaluation entry points
src/           Data, model, training and evaluation modules
reports/       Study protocol, amendments and feature-handoff documents
artifacts/
  figures/     Public aggregate figures
  metrics/     Public aggregate audits and results
  checkpoints/ Private ignored model checkpoints
  features/    Private ignored feature archives
joint_v2/      Separately contributed downstream classifier comparison
data/          Private ignored datasets and processed tables
```

## Notebooks

1. [SEER dataset and cohort](notebooks/01_seer_dataset_and_cohort.ipynb)
2. [Camelyon17 and joint model](notebooks/02_camelyon17_and_joint_model.ipynb)
3. [8D feature validation and handoff](notebooks/03_8d_feature_validation_and_handoff.ipynb)

## Installation

Python 3.12 was used for development.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

## Reproduction sequence

Run commands from the repository root.

### SEER preparation

```bash
python scripts/build_seer_v2_cohort.py
python scripts/build_seer_v2_splits.py
python scripts/build_seer_v2_model_table.py
python scripts/build_seer_v2_preprocessor.py
python scripts/analyze_seer_v2_drift.py
```

### Joint training

```bash
python scripts/train_joint_model_v2.py --seed 42
python scripts/train_joint_model_v2.py --seed 123
python scripts/train_joint_model_v2.py --seed 2026
python scripts/summarize_joint_v2_seeds.py
```

### Shared-feature extraction

Run both tasks for every seed:

```bash
python scripts/extract_joint_v2_features.py --seed 42 --task image
python scripts/extract_joint_v2_features.py --seed 42 --task tabular
```

Repeat for seeds 123 and 2026.

### PCA reduction

```bash
python scripts/build_joint_v2_pca_features.py --seed 42 --task image
python scripts/build_joint_v2_pca_features.py --seed 42 --task tabular
```

Repeat for seeds 123 and 2026.

### Representation validation

```bash
python scripts/evaluate_joint_v2_linear_probes.py
```

The final locked-test extraction and evaluation are intentionally excluded from the reproduction sequence. Their scripts and audits are retained for transparency, but the one-time final evaluation must not be rerun.

## Data access and privacy

Raw datasets are not distributed through this repository.

Camelyon17 must be obtained from an authorised public source and placed under the ignored `data/` directory.

SEER data requires an approved SEER data-access account and remains subject to the SEER Research Data Agreement. Raw SEER files, processed patient-level tables, category vocabularies, checkpoints and feature archives must not be committed.

Only privacy-safe aggregate audits, configurations, metrics and figures are tracked by Git.

## Final locked-test procedure

The following splits were kept locked throughout development:

- Camelyon `ood_test`.
- SEER `temporal_test`.

They were not used for preprocessing, tuning, calibration, threshold selection, checkpoint selection or development evaluation.

After the upstream and downstream procedures were frozen, the locked 8D representations were generated and evaluated once. File hashes, checkpoint hashes, row counts, dimensions and label-order preservation were audited.

The final results are now fixed. The locked-test evaluation must not be rerun, and no architecture, preprocessing, threshold, seed or model-selection decision may be changed in response to the results.

## Current status

Completed:

- Camelyon and SEER exploratory analysis.
- Leakage-safe SEER cohort and feature construction.
- Deterministic development and temporal splits.
- Three-seed joint-model training.
- Development-only threshold analysis.
- 32D shared-feature extraction.
- Training-only 32D-to-8D PCA reduction.
- Three-seed linear-probe validation.
- Privacy-safe encrypted development handoff.
- Audited extraction of all six locked-test 8D representations.
- Verification and freezing of all 12 downstream model states.
- One-time classical-versus-quantum locked-test evaluation.
- Public locked-test result report and aggregate JSON.
- Reproducible documentation notebooks.

Pending:

- Final five-page report.
- Final presentation.

## Limitations

- Camelyon and SEER contain different patients and prediction targets.
- This is multi-task representation learning, not patient-level multimodal fusion.
- SEER performance does not establish clinical usefulness.
- Camelyon class 0 does not mean the patient is cancer-free.
- The downstream model states required a documented development-only recovery because the original runner had not persisted them.
- The quantum circuit was simulated rather than executed on quantum hardware.
- The locked-test results do not replace validation on an additional independent external cohort.
- The project is a research prototype and not a medical diagnostic system.

## Project documents

- [Study protocol](reports/study_protocol.md)
- [Joint model v2 development selection](reports/model_v2_development_selection.md)
- [Feature handoff plan](reports/joint_v2_feature_handoff_plan.md)
- [8D validation report](reports/joint_v2_8d_validation.md)
- [Final locked-feature protocol](reports/final_locked_feature_protocol.md)
- [Final locked-test report](joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md)
- [Final locked-test aggregate results](joint_v2/results/locked_test/final_locked_test_results.json)
- [Downstream contributor documentation](joint_v2/README.md)
