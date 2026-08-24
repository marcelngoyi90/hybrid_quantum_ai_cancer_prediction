# Joint Model v2: Eight-Dimensional Representation Validation and Handoff

## 1. Status

This document records the completed development-stage validation of the shared representations produced by joint model v2.

The model was trained jointly on two unpaired cancer datasets:

- Camelyon17-WILDS histopathology images.
- SEER breast-cancer registry data.

The shared representation was extracted from the model, reduced from 32 dimensions to 8 dimensions, and evaluated using a fixed classical linear probe.

All results in this document are development results. The Camelyon `ood_test` and SEER `temporal_test` splits remain locked and have not been evaluated.

## 2. Prediction tasks

### 2.1 Camelyon17 image task

The input is a 96 × 96 histopathology patch taken from a lymph-node slide.

The binary target is:

- `0`: no metastatic tumour tissue in the labelled central region of the patch.
- `1`: metastatic tumour tissue present in the labelled central region.

A negative patch prediction does not mean that the patient is cancer-free. It only describes the evaluated patch region.

### 2.2 SEER tabular task

The input consists of 15 leakage-safe demographic and tumour-related variables from patients already diagnosed with breast cancer.

The binary target is:

- `0`: N0, meaning no positive regional lymph nodes were recorded.
- `1`: N+, meaning at least one positive regional lymph node was recorded.

The SEER model does not predict whether a person has breast cancer. Every person in this cohort already has a breast-cancer diagnosis. It predicts recorded regional lymph-node involvement.

## 3. Development data

### 3.1 Camelyon17-WILDS

| Split | Patches | Role |
|---|---:|---|
| Train | 302,436 | Model training and representation fitting |
| ID validation | 33,560 | Supporting in-distribution evaluation |
| OOD validation | 34,904 | Primary development evaluation |
| OOD test | 85,054 | Locked final evaluation |

The OOD validation and test splits come from hospitals excluded from the training hospitals. This provides a direct evaluation of generalisation under hospital and staining distribution shift.

### 3.2 SEER

The adult index cohort contains 731,278 patients:

- N0: 503,794 patients.
- N+: 227,484 patients.
- N+ prevalence: 31.11%.

| Split | Patients | Role |
|---|---:|---|
| Train | 461,292 | Model training and representation fitting |
| ID validation | 51,241 | Internal validation |
| Temporal validation | 107,249 | Development evaluation using diagnoses from 2020–2021 |
| Temporal test | 111,496 | Locked evaluation using diagnoses from 2022–2023 |

Patient membership is mutually exclusive across splits. The temporal test split has not been loaded during model development, PCA fitting, threshold selection or linear-probe evaluation.

## 4. Joint model

The model uses two modality-specific input paths:

- DenseNet121 processes Camelyon image patches.
- A neural tabular encoder processes SEER variables.

Both paths project their inputs into a 128-dimensional space. They then use the same shared trunk:

\[
128 \rightarrow 64 \rightarrow 32
\]

Separate binary classification heads produce the image and tabular outputs.

Training alternates one Camelyon mini-batch and one SEER mini-batch during every joint optimisation step. Both task losses have equal weight.

This is hard-parameter-sharing multi-task learning on unpaired datasets. It is not patient-level multimodal fusion, because the Camelyon and SEER records do not describe the same individuals.

## 5. Selected checkpoints

The same architecture was trained with three predeclared random seeds.

Checkpoint selection used:

\[
0.5 \times \text{Camelyon OOD validation ROC-AUC}
+
0.5 \times \text{SEER temporal validation ROC-AUC}
\]

| Seed | Selected epoch | Joint validation score |
|---:|---:|---:|
| 42 | 2 | 0.885301 |
| 123 | 5 | 0.889257 |
| 2026 | 2 | 0.885704 |

The mean joint checkpoint score was:

\[
0.886754 \pm 0.002177
\]

This score is the mean of two ROC-AUC values. It is not an accuracy measurement.

No seed was selected simply because it produced the highest result. All three seeds were retained for robustness analysis.

## 6. Joint-model development performance

Results are reported as mean ± sample standard deviation across the three seeds.

| Task | Development split | Accuracy | ROC-AUC |
|---|---|---:|---:|
| Camelyon image task | OOD validation | 0.928394 ± 0.011533 | 0.978926 ± 0.002913 |
| SEER tabular task | Temporal validation | 0.759936 ± 0.001330 | 0.794581 ± 0.001608 |

The image model therefore achieved approximately 92.84% OOD validation accuracy and 97.89% OOD ROC-AUC.

The lower SEER accuracy should not be interpreted as a failed model. The SEER task is a different, patient-level clinical prediction problem with incomplete and overlapping risk information. ROC-AUC, sensitivity, specificity, balanced accuracy and calibration must be considered alongside overall accuracy.

## 7. SEER operating threshold

The default classification threshold of 0.5 produced high specificity but relatively low N+ sensitivity.

Thresholds were therefore selected separately for each seed using only SEER `id_val`. The temporal validation split was used only to evaluate the already-selected thresholds.

The mean selected threshold was:

\[
0.311 \pm 0.011
\]

Temporal-validation results were:

| Metric | Threshold 0.5 | ID-validation-selected threshold |
|---|---:|---:|
| Accuracy | 0.759936 | 0.702639 |
| Balanced accuracy | 0.671293 | 0.714732 |
| F1 | 0.529579 | 0.607809 |
| Sensitivity | 0.439559 | 0.746346 |
| Specificity | 0.903026 | 0.683118 |

The lower threshold substantially increased the detection of N+ patients, but it also produced more false-positive N+ predictions. It changed the operating trade-off; it did not change ROC-AUC or the underlying discrimination of the model.

Both the default and balanced operating points should be reported.

These thresholds belong to the original joint SEER head. They were not automatically transferred to the separate 8D logistic probe.

## 8. Shared feature extraction

For every selected checkpoint, the output of the shared 32-dimensional trunk was extracted before the task-specific classification head.

The extraction was completed separately for:

- Seeds 42, 123 and 2026.
- Camelyon and SEER.
- Their permitted training and development-validation splits.

The output archives contain:

- A floating-point feature matrix.
- A binary label vector.

They do not contain:

- Patient identifiers.
- Raw SEER clinical variables.
- Camelyon hospital, patient or slide identifiers.
- Original images.
- Joint-model probabilities.
- Locked-test records.

## 9. Reduction from 32 dimensions to 8 dimensions

A separate dimensionality-reduction pipeline was fitted for every task and seed, producing six pipelines.

Each pipeline used:

1. `StandardScaler` fitted on the relevant training representation.
2. PCA with exactly eight components fitted on the scaled training representation.
3. No labels during scaler or PCA fitting.
4. No validation or locked-test rows during fitting.

The fitted training pipeline was then reused to transform the permitted validation splits.

### 9.1 Explained variance

| Seed | Task | Total variance retained by 8 components |
|---:|---|---:|
| 42 | Camelyon | 0.999635 |
| 42 | SEER | 0.999425 |
| 123 | Camelyon | 0.999737 |
| 123 | SEER | 0.999728 |
| 2026 | Camelyon | 0.999468 |
| 2026 | SEER | 0.999130 |

All six pipelines retained more than 99.9% of the variance in their corresponding 32-dimensional training representations.

Explained variance alone does not demonstrate that target-related information was preserved. For that reason, predictive validation was performed separately.

## 10. Predictive validation of the 8D representations

The same unregularised logistic-regression probe was fitted to both representations:

- Original shared 32D features.
- PCA-reduced 8D features.

For each seed and task:

- Training used only the corresponding training split.
- Evaluation used only permitted development splits.
- The 32D and 8D probes received identical rows and labels.
- No hyperparameter tuning was performed.
- No probability calibration or threshold optimisation was performed.
- The classification threshold remained 0.5.
- No locked-test split was loaded.

A total of 12 probes were fitted:

\[
3\text{ seeds} \times
2\text{ tasks} \times
2\text{ representations}
= 12
\]

### 10.1 Camelyon OOD validation

| Representation | Accuracy | ROC-AUC |
|---|---:|---:|
| Shared 32D | 0.932988 ± 0.008783 | 0.980280 ± 0.003992 |
| PCA 8D | 0.929540 ± 0.007283 | 0.978598 ± 0.002986 |

The mean 8D-minus-32D ROC-AUC difference was:

\[
-0.001683 \pm 0.001305
\]

Mean ROC-AUC retention was:

\[
0.998287 \pm 0.001328
\]

The Camelyon 8D representation retained approximately 99.83% of the 32D linear-probe ROC-AUC.

### 10.2 SEER temporal validation

| Representation | Accuracy | ROC-AUC |
|---|---:|---:|
| Shared 32D | 0.760526 ± 0.000851 | 0.794814 ± 0.001372 |
| PCA 8D | 0.760361 ± 0.000944 | 0.794643 ± 0.001563 |

The mean 8D-minus-32D ROC-AUC difference was:

\[
-0.000172 \pm 0.000235
\]

Mean ROC-AUC retention was:

\[
0.999784 \pm 0.000296
\]

The SEER 8D representation retained approximately 99.98% of the 32D linear-probe ROC-AUC.

## 11. Interpretation

The eight-dimensional representations preserve nearly all linearly accessible predictive information found in the original 32-dimensional shared representations.

The 8D linear probes also closely reproduce the development ROC-AUC of the original joint task heads:

- Camelyon 8D probe ROC-AUC: 0.978598.
- Original Camelyon joint head ROC-AUC: 0.978926.
- SEER 8D probe ROC-AUC: 0.794643.
- Original SEER joint head ROC-AUC: 0.794581.

Small differences are expected because the original task heads and the independently fitted logistic probes are not identical classifiers.

These results support using the 8D representations for the downstream classical-versus-quantum comparison.

They do not show that a quantum classifier is better than a classical classifier. No quantum result has been produced at this stage.

## 12. Handoff interface

For every task, split and seed, the private archive contains exactly two arrays:

- `features`: an `N × 8` `float32` matrix.
- `labels`: an `N`-element `uint8` binary vector.

### 12.1 Camelyon packages per seed

| Split | Shape |
|---|---:|
| Train | 302,436 × 8 |
| ID validation | 33,560 × 8 |
| OOD validation | 34,904 × 8 |

### 12.2 SEER packages per seed

| Split | Shape |
|---|---:|
| Train | 461,292 × 8 |
| ID validation | 51,241 × 8 |
| Temporal validation | 107,249 × 8 |

The quantum teammate must preserve:

- Feature order.
- Label order.
- Training and validation membership.
- Task definitions.
- Seed identity.
- Evaluation metrics.
- Locked-test protection.

The Camelyon and SEER tasks must be evaluated separately because their labels describe different units and outcomes.

## 13. Classical-versus-quantum fairness

For a given task, split and seed, the classical and quantum classifiers must receive the exact same feature matrix and label vector.

If the quantum implementation requires additional scaling or angle mapping:

- It must be fitted or defined using training data only.
- It must not use locked-test information.
- The same transformed inputs must be provided to the final classical comparator.
- The transformation must be documented and frozen before final testing.

The current logistic probe is a validated classical reference. If the quantum pipeline introduces an additional input transformation, the final classical comparator must be rerun using that same transformation.

Results must be reported for all three seeds using individual results and mean ± sample standard deviation. The best-looking seed must not be reported alone.

## 14. Privacy and data-use restrictions

SEER-derived feature matrices remain patient-level derived data even though identifiers and raw clinical fields were removed.

Therefore:

- SEER feature archives must remain outside Git.
- Fitted SEER preprocessing and probe states must remain outside Git.
- SEER archives must not be sent to another person until the applicable SEER data-use agreement and that person’s authorisation have been confirmed.
- Until authorisation is confirmed, the teammate may build the classifier interface using synthetic `N × 8` data.
- Public reports may contain only aggregate metrics, hashes, shapes and reproducibility information.

The committed repository contains public aggregate audits but not private feature values, labels, probabilities, predictions or patient identifiers.

## 15. Locked-test rule

The following splits remain locked:

- Camelyon `ood_test`.
- SEER `temporal_test`.

They must remain locked until the following are frozen:

- Quantum circuit architecture.
- Quantum feature-encoding procedure.
- Classical comparator configuration.
- Any additional training-only feature scaling.
- Optimisers and training schedules.
- Model-selection metric.
- Threshold rule.
- Three-seed aggregation procedure.
- Output and audit format.

After these decisions are frozen, the existing joint checkpoints and PCA pipelines will transform the locked splits without refitting.

Classical and quantum models will then be evaluated once on identical locked-test representations. No model, threshold or preprocessing decision may be changed after viewing those results.

## 16. Current conclusion

Joint model v2 produced stable development performance across three seeds and two clinically related lymph-node prediction tasks.

The shared 32-dimensional representations were successfully reduced to eight dimensions while retaining:

- More than 99.9% of representation variance.
- Approximately 99.83% of Camelyon linear-probe ROC-AUC.
- Approximately 99.98% of SEER linear-probe ROC-AUC.

The eight-dimensional development features are technically ready for downstream classifier work.

The next responsibility of the current contributor is documentation and privacy-safe handoff. Quantum classifier implementation belongs to the quantum teammate.