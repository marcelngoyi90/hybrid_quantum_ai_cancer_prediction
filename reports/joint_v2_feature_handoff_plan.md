# Joint Model v2 Feature Handoff Plan

## 1. Purpose

This document defines how the completed joint model v2 will produce features for the downstream classical-versus-quantum comparison.

The downstream stage uses the representations learned jointly from Camelyon17-WILDS and SEER. It does not use a paired dataset or BCNB.

The purpose of this plan is to prevent:

- Selection of only the best-looking random seed.
- Test-set leakage.
- Different preprocessing for the classical and quantum classifiers.
- Exposure of SEER patient identifiers or raw patient-level records.

## 2. Completed joint-model development

The same joint architecture was trained with three predeclared random seeds:

- Seed 42.
- Seed 123.
- Seed 2026.

The selected checkpoint for each seed was determined using the frozen joint validation score:

\[
0.5 \times \text{Camelyon OOD validation ROC-AUC}
+
0.5 \times \text{SEER temporal validation ROC-AUC}
\]

The selected checkpoints are:

| Seed | Selected epoch | Joint validation score |
|---:|---:|---:|
| 42 | 2 | 0.885301 |
| 123 | 5 | 0.889257 |
| 2026 | 2 | 0.885704 |

No locked test split was used for preprocessing, tuning, checkpoint selection or model evaluation.

## 3. Three-seed development results

The primary development results are reported across all three seeds.

| Task | Validation split | Mean accuracy | Mean ROC-AUC |
|---|---|---:|---:|
| Camelyon17 image task | OOD validation | 0.928394 | 0.978926 |
| SEER tabular task | Temporal validation | 0.759936 | 0.794581 |

The mean joint checkpoint score is:

\[
0.886754 \pm 0.002177
\]

This joint score is an average of two ROC-AUC values. It is not an accuracy value.

## 4. Cross-seed rule

No single random seed will be selected because it produced the highest development metric.

All three selected checkpoints will be retained as robustness repetitions of the same model architecture.

Feature extraction and the classical-versus-quantum comparison will be performed separately for seeds 42, 123 and 2026.

Results will be reported using:

- The individual result from every seed.
- The mean across the three seeds.
- The sample standard deviation across the three seeds.

This rule prevents post-hoc selection of the most favorable random seed.

## 5. Source representation

For each seed, features will be extracted from the output of the shared trunk immediately before the task-specific classification head.

The extracted shared representation has 32 dimensions.

The following will not be used as quantum or classical classifier inputs:

- Final task logits.
- Predicted probabilities.
- Patient identifiers.
- Camelyon hospital or center metadata.
- SEER split labels.
- Raw target-generating variables.
- Any locked-test information.

The joint model will be placed in evaluation mode during extraction. Training augmentation and dropout will be disabled.

## 6. Development feature splits

Features will first be extracted only from development splits.

### Camelyon17

- Training features: `train`.
- Internal validation features: `id_val`.
- Distribution-shift validation features: `ood_val`.
- Locked split excluded: `ood_test`.

### SEER

- Training features: `train`.
- Internal validation features: `id_val`.
- Temporal validation features: `temporal_val`.
- Locked split excluded: `temporal_test`.

The locked splits will not be loaded during feature-pipeline development.

## 7. Reduction from 32 dimensions to 8 dimensions

The 32-dimensional representation will be reduced to 8 dimensions separately for each task and random seed.

This produces six fitted reduction pipelines:

- Three Camelyon reduction pipelines.
- Three SEER reduction pipelines.

Each reduction pipeline will contain:

1. A standard scaler fitted on the corresponding training features.
2. PCA with exactly eight components fitted on the scaled training features.

The scaler and PCA model must be fitted using the training split only.

Validation and eventual locked-test features will only be transformed using the already-fitted training scaler and PCA model.

Labels will not be used when fitting PCA.

The following information will be recorded for each fitted PCA model:

- Random seed.
- Task.
- Number of training examples.
- Original dimension.
- Reduced dimension.
- Explained variance ratio for every component.
- Total explained variance.
- Fitted-state hash.
- Input-checkpoint hash.

PCA hyperparameters will not be selected using a locked test split.

## 8. Eight-dimensional feature packages

Each private feature package will contain only:

- An `N × 8` floating-point feature matrix.
- An `N`-element binary label vector.
- The task name.
- The split name.
- The originating random seed.
- The feature-pipeline version.

The packages will not contain:

- SEER Patient ID.
- Raw SEER clinical variables.
- Category vocabularies.
- Camelyon patient, slide or hospital identifiers.
- Original images.
- Predicted probabilities from the joint model.

SEER-derived row-level feature packages must remain private and must not be committed to Git.

Before sending SEER-derived row-level features to another person, the applicable SEER data-use agreement and that person’s authorization must be confirmed. Until then, the teammate may develop the classifier interface using synthetic `N × 8` data.

## 9. Classical-versus-quantum fairness rule

For a given task, split and seed, the classical and quantum classifiers must receive the exact same 8-dimensional feature matrix and binary labels.

They must also use the same:

- Training rows.
- Validation rows.
- Locked-test rows during final evaluation.
- Feature order.
- Target definition.
- Development split.
- Evaluation metrics.
- Model-selection information.

Neither classifier may receive an additional clinical variable, image feature or metadata field unavailable to the other classifier.

Any scaling required for quantum encoding must be fitted on the training split only and must also be supplied to the classical comparator.

## 10. Classifier development

The classical and quantum classifiers will be developed using training data and the permitted validation splits only.

### Camelyon17 model selection

- Train on `train`.
- Use `ood_val` as the main model-selection split.
- Retain `id_val` for supporting in-distribution analysis.
- Do not use `ood_test` during development.

### SEER model selection

- Train on `train`.
- Use `temporal_val` as the main model-selection split.
- Retain `id_val` for supporting internal analysis.
- Do not use `temporal_test` during development.

The primary comparison metric will be ROC-AUC.

Supporting metrics will include:

- Accuracy.
- Balanced accuracy.
- Average precision.
- F1 score.
- Sensitivity.
- Specificity.
- Precision.
- Brier score, when valid probabilities are available.

## 11. Locked-test rule

Camelyon `ood_test` and SEER `temporal_test` remain locked until all of the following are frozen:

- The 32D feature extraction implementation.
- The training-only scaler.
- The eight-component PCA implementation.
- The classical classifier architecture.
- The quantum classifier architecture.
- All optimizer and training settings.
- The model-selection metric.
- The classification-threshold rule.
- The three-seed aggregation procedure.
- The output and audit formats.

After these decisions are frozen, one final evaluation procedure will evaluate all three seed pipelines on the locked splits.

No model, threshold, PCA pipeline or hyperparameter may be changed after viewing locked-test results.

## 12. Required outputs

The feature-handoff stage will produce:

1. A private 32-dimensional feature package for each task, split and seed.
2. A private 8-dimensional feature package for each task, split and seed.
3. A private fitted scaler and PCA state for each task and seed.
4. A public feature-extraction audit containing no row-level data.
5. A public PCA explained-variance report.
6. A reproducibility script for feature extraction.
7. A reproducibility script for the 32D-to-8D transformation.
8. A documented interface for the classical and quantum classifiers.

Large feature matrices and fitted private preprocessing states must remain excluded from Git.

## 13. Execution order

The implementation order is:

1. Validate the three selected checkpoints.
2. Build deterministic 32-dimensional feature extraction.
3. Extract development features for all three seeds.
4. Audit feature shapes, labels and split protection.
5. Fit the six training-only scaler and PCA pipelines.
6. Produce the 8-dimensional development features.
7. Provide the fixed classifier interface to the quantum teammate.
8. Develop the classical and quantum classifiers.
9. Freeze every downstream decision.
10. Run the final locked-test evaluation once.
11. Report individual-seed and mean ± standard-deviation results.

This plan governs the downstream feature handoff for joint model v2.
