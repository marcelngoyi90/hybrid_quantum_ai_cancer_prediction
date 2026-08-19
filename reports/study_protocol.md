# Hybrid Quantum AI for Cancer Risk Prediction
## Study Protocol

Version: 0.1  
Status: Frozen before SEER model training

## 1. Purpose

This project investigates whether two large, unpaired cancer datasets from different modalities can learn a useful shared representation.

The project combines:

1. Histopathology images from Camelyon17-WILDS.
2. Clinical and registry variables from SEER breast cancer data.
3. A paired image-and-clinical dataset, provisionally BCNB, for transfer and fusion.
4. A later classical-versus-quantum comparison using the same 8-dimensional representation.

The datasets will not be joined row by row because they do not contain the same patients.

## 2. Main research questions

### RQ1 — Image model

Can a fine-tuned DenseNet121 identify metastatic tumor tissue in lymph-node image patches and generalize to an unseen hospital?

### RQ2 — Tabular model

Can clinical and tumor characteristics predict whether a diagnosed breast cancer case has positive regional lymph nodes?

### RQ3 — Multi-task learning

Can unpaired image and tabular datasets train a shared neural representation without causing unacceptable negative transfer?

### RQ4 — Paired transfer

Do the image and tabular encoders learned from Camelyon17 and SEER improve performance when transferred to a paired image-and-clinical dataset?

### RQ5 — Quantum comparison

Can a quantum classifier perform competitively with a classical classifier when both receive exactly the same 8-dimensional features?

## 3. Dataset tasks

### 3.1 Camelyon17-WILDS

Input:

- A 96 × 96 histopathology image patch from a lymph node.

Target:

- 0: No metastatic tumor tissue in the patch.
- 1: Metastatic tumor tissue is present in the patch.

This output applies only to the examined patch. It does not mean that the patient is cancer-free.

Available splits:

| Split | Patches | Hospitals |
|---|---:|---|
| Train | 302,436 | 0, 3 and 4 |
| ID validation | 33,560 | 0, 3 and 4 |
| OOD validation | 34,904 | 1 |
| OOD test | 85,054 | 2 |

The OOD splits measure generalization to hospitals not used for training.

### 3.2 SEER breast cancer dataset

Input:

- Demographic and clinicopathological variables from diagnosed female breast cancer cases.

Target:

- 0 — N0: No positive regional lymph nodes recorded.
- 1 — N+: One or more positive regional lymph nodes recorded.

Target construction:

- Code 00 becomes N0.
- Codes 01–90, 95 and 97 become N+.
- Unknown, invalid or unusable target values are excluded.

This is not a general cancer diagnosis prediction. Every record already represents a diagnosed breast cancer case. The model predicts regional lymph-node involvement.

### 3.3 Relationship between the targets

The two targets are related but not identical.

- Camelyon17 detects metastatic tissue at image-patch level.
- SEER predicts regional lymph-node positivity at patient/tumor-record level.

Their shared clinical concept is lymph-node metastasis. Their different modalities and prediction units are part of the research question.

## 4. SEER cohort and leakage control

The primary SEER cohort will use one deterministic index breast cancer record per patient.

Index record selection:

1. Select the earliest eligible diagnosis year.
2. If multiple records remain, use the earliest valid sequence number.
3. Record and report how many duplicate and conflicting patient records were removed.

Primary predictors may include:

- Age.
- Race.
- Hispanic origin.
- Primary breast site.
- Histology.
- Laterality.
- Tumor size.
- Unified tumor grade.
- Estrogen receptor status.
- Progesterone receptor status.
- HER2 status.
- Breast cancer subtype.

The following variables will not be model inputs:

- Patient ID.
- Regional nodes positive.
- Regional nodes examined.
- Year of diagnosis.
- Diagnostic confirmation.
- Any variable created directly from the target.

Patient ID is used only for deduplication and leakage-safe splitting.

Year of diagnosis is used only for temporal splitting and subgroup analysis.

The older and newer SEER grade columns will be combined into one unified grade variable. Structural missing values caused by changes in SEER coding years will not be treated as ordinary missing clinical data.

## 5. SEER data splits

The primary SEER evaluation will measure temporal generalization:

- Development period: 2010–2019.
- Temporal validation: 2020–2021.
- Locked temporal test: 2022–2023.

The development period will contain:

- Training patients.
- An internal validation subset drawn only from 2010–2019.

No patient may occur in more than one split.

The exact counts and class distributions must be recorded before model training.

If the index-record construction makes any temporal split too small or unrepresentative, the protocol will be amended before training, with the reason documented.

## 6. Models

### 6.1 Image baseline

The primary image model will be:

- ImageNet-pretrained DenseNet121.
- All DenseNet layers fine-tuned.
- A binary classification head.

The pilot comparison of frozen, last-block, full fine-tuning and training from scratch identified full fine-tuning as the strongest candidate.

The pilot is exploratory because it used:

- 20,000 training patches.
- Five epochs.
- One random seed.
- OOD validation accuracy for checkpoint selection.

The pilot results are not final results.

The final image model will use the complete Camelyon17 training split and will select checkpoints using OOD-validation AUROC.

### 6.2 Tabular baselines

Three standalone tabular model families will be compared:

1. CatBoost.
2. A gradient-boosted tree model such as XGBoost or LightGBM.
3. A neural tabular multilayer perceptron.

The tree models provide strong tabular baselines.

The neural model is required because its encoder can participate in hard-parameter-sharing and later transfer.

### 6.3 Hard-parameter-sharing model

The multi-task model will contain:

Image path:

- DenseNet121 image encoder.
- Image projector to a 128-dimensional vector.

Tabular path:

- Tabular neural encoder.
- Tabular projector to a 128-dimensional vector.

Shared path:

- Shared layer: 128 to 64 dimensions.
- Shared layer: 64 to 32 dimensions.

Task-specific outputs:

- Image head: patch tumor absent or present.
- SEER head: N0 or N+.

The two datasets are not concatenated and are not treated as paired samples.

One training cycle will:

1. Process one Camelyon17 mini-batch.
2. Process one SEER mini-batch.
3. Calculate one normalized loss for each task.
4. Combine the two task losses.
5. Update the shared and task-specific parameters.

The initial loss will be:

L_total = 0.5 × L_image + 0.5 × L_tabular

Batch losses will be averaged within each modality so that the larger tabular batch does not automatically dominate the shared layers.

## 7. Gradient-conflict analysis

For the shared layers, training will record:

- Image-task gradient norm.
- Tabular-task gradient norm.
- Cosine similarity between the two task gradients.
- Individual task losses.
- Combined loss.

Negative cosine similarity indicates that the two tasks are requesting conflicting changes to the shared parameters.

The first multi-task model will use ordinary equal-weight training.

Methods such as dynamic loss weighting or PCGrad will be tested only as documented secondary experiments if the baseline shows gradient conflict or negative transfer.

This analysis is part of the project’s distinctive contribution.

## 8. Evaluation protocol

### Primary metric

- ROC-AUC.

### Secondary metrics

- Average precision or PR-AUC.
- Balanced accuracy.
- Accuracy.
- F1 score.
- Precision.
- Sensitivity.
- Specificity.
- Brier score.
- Calibration error.
- Confusion matrix.

Accuracy will not be used as the only success measure.

A probability threshold of 0.5 will always be reported.

If a second operating threshold is selected, it must be selected using validation data only and then frozen before test evaluation.

### Repeated experiments

Final experiments will use at least three seeds:

- 42.
- 123.
- 2026.

Results will be reported as mean and standard deviation across seeds.

Where practical, 95% bootstrap confidence intervals will be calculated:

- By patient for SEER.
- By patient or slide for Camelyon17.
- By patient for the paired dataset.

## 9. Negative-transfer rule

For each task:

Delta AUROC = multi-task AUROC − standalone AUROC

Hard parameter sharing will be considered harmful to a task if its mean validation AUROC decreases by more than 0.02 compared with its standalone model.

The result will not be hidden if multi-task learning performs worse.

A neutral or negative multi-task result is still scientifically meaningful because it tests whether unrelated modalities actually benefit from shared parameters.

## 10. Test-set protection

### Camelyon17

- OOD validation hospital 1 may be used for model selection.
- OOD test hospital 2 remains locked until the final image and multi-task configurations are selected.

### SEER

- The 2022–2023 temporal test split remains locked until the final preprocessing and model configurations are selected.

### Paired dataset

- Splits must be patient-level.
- No slide, patch or clinical record belonging to a test patient may appear during training.

No hyperparameter, feature, checkpoint or threshold will be selected using a test set.

## 11. Paired transfer and fusion

The paired-data phase is provisional until BCNB is downloaded and audited.

For each paired patient:

Image branch:

1. Divide the whole-slide image into tissue patches.
2. Encode each patch using the transferred DenseNet121.
3. Combine patch features using attention-based multiple-instance learning.
4. Produce one patient-level image vector.

Clinical branch:

1. Select clinical features available in both SEER and the paired dataset.
2. Apply the transferred tabular encoder.
3. Produce one patient-level clinical vector.

Fusion:

1. Pass both vectors through their learned shared representation.
2. Concatenate the image and clinical vectors.
3. Apply a fusion multilayer perceptron.
4. Produce an 8-dimensional bottleneck.
5. Predict N0 versus N+.

Initial paired training will freeze the transferred encoders.

Later fine-tuning may unfreeze:

- The final DenseNet block.
- Modality projectors.
- Shared layers.

Small learning rates will be used during unfreezing.

## 12. Paired-transfer ablations

The following comparisons will be performed:

1. Image-only paired model.
2. Clinical-only paired model.
3. Paired fusion trained without Camelyon17 or SEER transfer.
4. ImageNet image transfer plus paired clinical training.
5. Camelyon17 image transfer plus SEER tabular transfer.
6. Unpaired multi-task shared-transfer model.

These comparisons are necessary to determine whether the proposed pretraining actually contributes value.

## 13. Classical and quantum handoff

The final fusion model will export:

- One 8-dimensional feature vector per patient.
- Patient-level target.
- Split assignment.
- Classical prediction probability.
- Relevant non-identifying metadata.
- Reproduction instructions.

The classical and quantum classifiers must use:

- Exactly the same 8-dimensional features.
- Exactly the same train, validation and test patients.
- The same primary metrics.
- The same test-set restrictions.

The quantum model will not receive privileged data unavailable to the classical comparator.

## 14. Distinctive contribution

The project does not claim that multi-task learning or quantum classification is new by itself.

The distinctive experiment is the combined evaluation of:

- Two large unpaired cancer datasets.
- Two different modalities.
- A shared representation centered on lymph-node metastasis.
- Hospital distribution shift in histopathology.
- Temporal distribution shift in registry data.
- Explicit measurement of task-gradient conflict.
- Transfer into a paired image-and-clinical dataset.
- A controlled classical-versus-quantum comparison.

Novelty will be demonstrated through experiments and ablations, not claimed from architecture names alone.

## 15. Reporting rules

The project will:

- Separate exploratory pilot results from final results.
- Report all planned standalone and multi-task comparisons.
- Report unsuccessful experiments.
- Clearly distinguish patch-level, tumor-record-level and patient-level predictions.
- Avoid statements such as “cancer-free” when the model evaluates only a patch or lymph-node status.
- Avoid presenting the system as a clinical diagnostic tool.
- Document all protocol amendments before the affected experiment is trained.

## 16. Execution order

1. Freeze this study protocol.
2. Implement reproducible SEER EDA and cohort construction.
3. Generate and freeze SEER patient splits.
4. Train standalone SEER baselines.
5. Retrain the final Camelyon17 baseline on the complete training data.
6. Train the basic hard-parameter-sharing model.
7. Measure gradient conflict and negative transfer.
8. Run justified multi-task ablations.
9. Download and audit the paired dataset.
10. Train paired image, clinical and fusion baselines.
11. Transfer the learned encoders and shared representation.
12. Export the fixed 8-dimensional features.
13. Perform the classical-versus-quantum comparison.
14. Evaluate each locked test set once.
15. Prepare the final report and presentation.