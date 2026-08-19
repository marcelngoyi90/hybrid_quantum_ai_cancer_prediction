# Hybrid Quantum AI for Cancer Risk Prediction

## Study Protocol

Version: 1.0.0
Status: Frozen before joint-model training

This project trains one joint neural network using two large, unpaired cancer datasets with different modalities. The network contains modality-specific encoders, shared representation layers and two task-specific outputs.

## 1. Purpose

This project investigates whether one neural network can learn a useful shared representation from two large, unpaired cancer datasets with different modalities.

The project uses:

1. Histopathology images from Camelyon17-WILDS.
2. Clinical and registry variables from SEER breast cancer data.
3. A paired image-and-clinical dataset, provisionally BCNB, for later transfer and fusion.
4. A later classical-versus-quantum comparison using the same fixed 8-dimensional representation.

The Camelyon17 and SEER datasets will not be joined row by row because they do not contain the same patients.

Instead, one joint model will learn from alternating image and tabular mini-batches.

## 2. Main research questions

### RQ1 — Image capability

Can the image branch of one joint model identify metastatic tumor tissue in lymph-node image patches and generalize to an unseen hospital?

### RQ2 — Tabular capability

Can the tabular branch of the same joint model predict whether a diagnosed breast cancer case has positive regional lymph nodes?

### RQ3 — Joint learning

Can one neural network learn from large, unpaired image and tabular cancer datasets through modality-specific encoders and a shared representation while maintaining useful performance on both tasks?

### RQ4 — Paired transfer

Can the image encoder, tabular encoder and shared representation learned from Camelyon17 and SEER be transferred to a paired image-and-clinical dataset?

### RQ5 — Quantum comparison

Can a quantum classifier perform competitively with a classical classifier when both receive exactly the same 8-dimensional features?

## 3. Dataset tasks

### 3.1 Camelyon17-WILDS

Input:

- A 96 × 96 histopathology image patch from a lymph node.

Target:

- 0: No metastatic tumor tissue is present in the patch.
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
- SEER predicts regional lymph-node positivity at patient or tumor-record level.

Their shared clinical concept is lymph-node metastasis.

Their different modalities and prediction units are part of the research question.

The joint model will therefore produce two different outputs rather than one combined cancer probability.

## 4. SEER cohort and leakage control

The primary SEER cohort uses one deterministic index breast cancer record per patient.

### Protocol amendment 0.1.1 — Adult primary cohort

Date: 2026-08-19

The initial EDA identified 12 records with age below 18 among 756,356 records. These consisted of 10 N0 and 2 N+ records.

The primary SEER cohort includes patients aged 18 or older. The age restriction was applied before selecting one index record per patient.

Rationale:

- The intended prediction population is adults with diagnosed breast cancer.
- The under-18 subgroup is too small for meaningful training or subgroup evaluation.
- The decision was made before cohort construction or model training.
- No model result was used to make this decision.

Index-record selection:

1. Select the earliest eligible diagnosis year.
2. If multiple records remain, use the earliest valid sequence number.
3. Remove subsequent eligible cancer records from the primary patient-level cohort.
4. Record all exclusions and decisions in the cohort audit.

The final adult index cohort contains 731,278 unique patients.

Primary model inputs are:

Numeric features:

- Age.
- Tumor size in millimetres.

Categorical features:

- Race.
- Hispanic origin.
- Primary breast site.
- Histologic type.
- Laterality.
- Unified tumor grade.
- Estrogen receptor status.
- Progesterone receptor status.
- HER2 status.
- Breast cancer subtype.
- Tumor-size status.

The following variables will not be model inputs:

- Patient ID.
- Sequence number.
- Regional nodes positive.
- Regional nodes examined.
- Year of diagnosis.
- Diagnostic confirmation.
- Raw grade columns replaced by unified grade.
- Raw age replaced by numeric age.
- Raw tumor-size code replaced by derived tumor-size features.
- Internal cohort-construction variables.
- Split assignment.
- Any variable created directly from the target.

Patient ID is used only for deduplication, cohort validation and leakage-safe splitting.

Year of diagnosis is used only for temporal splitting and aggregate auditing.

The older and newer SEER grade columns are combined into one unified grade variable. Structural missing values caused by SEER coding changes are not treated as ordinary missing clinical data.

## 5. SEER data splits

The primary SEER evaluation measures temporal generalization.

| Split | Years | Patients | N+ percentage |
|---|---|---:|---:|
| Train | 2010–2019 | 461,292 | 31.3938% |
| ID validation | 2010–2019 | 51,241 | 31.3889% |
| Temporal validation | 2020–2021 | 107,249 | 30.8739% |
| Locked temporal test | 2022–2023 | 111,496 | 30.0199% |

The internal validation subset was deterministically sampled within year-and-target strata from the 2010–2019 development period.

No patient occurs in more than one split.

The locked temporal test split cannot be used for feature selection, preprocessing decisions, hyperparameter tuning, checkpoint selection, calibration fitting or threshold selection.

Development drift analysis compares:

- Train against ID validation.
- Train against temporal validation.

The development drift analysis does not load the locked temporal test split.

## 6. Single joint model

The project will train one neural network with two modality-specific input paths, one shared representation and two task-specific outputs.

### 6.1 Image path

The image path will contain:

- An ImageNet-pretrained DenseNet121 encoder.
- Full DenseNet fine-tuning during joint training.
- A projector from the DenseNet representation to 128 dimensions.

The earlier comparison of frozen DenseNet, final-block fine-tuning, full fine-tuning and training from scratch was exploratory.

The pilot used:

- 20,000 training patches.
- Five epochs.
- One random seed.
- OOD-validation accuracy for checkpoint selection.

The pilot identified full fine-tuning as the strongest candidate. The pilot results are not final project results.

### 6.2 Tabular path

The tabular path will contain:

- Train-fitted preprocessing for two numeric features.
- Learned embeddings for eleven categorical features.
- A neural tabular encoder.
- A projector to 128 dimensions.

Numeric preprocessing will be fitted on the SEER training split only.

Numeric missing-value replacement and scaling statistics must not be learned from validation or test data.

Categorical vocabularies will be constructed from the SEER training split only.

Categories not observed during training will be mapped to a reserved unknown-category index.

### 6.3 Shared path

The projected image and tabular representations will enter the same shared layers:

- Shared layer: 128 to 64 dimensions.
- Shared layer: 64 to 32 dimensions.

The resulting 32-dimensional representation will be used by the appropriate task-specific output head.

### 6.4 Task-specific outputs

The joint model will have two output heads:

- Image head: metastatic tumor absent or present in the image patch.
- SEER head: N0 or N+ regional lymph-node status.

The outputs will always be reported separately.

The model will not produce one universal cancer diagnosis probability.

### 6.5 Alternating-batch training

The Camelyon17 and SEER datasets will not be concatenated or treated as paired patients.

One joint training cycle will:

1. Clear the accumulated gradients.
2. Process one Camelyon17 mini-batch.
3. Calculate the mean image-task binary loss.
4. Process one SEER mini-batch.
5. Calculate the mean tabular-task binary loss.
6. Combine the two losses.
7. Update the modality-specific encoders, shared layers and task-specific heads.

The initial combined loss will be:

L_total = 0.5 × L_image + 0.5 × L_tabular

Each task loss will be averaged within its own mini-batch before the losses are combined.

This prevents the larger tabular mini-batch from automatically dominating the shared parameters.

## 7. Joint-training monitoring

Training will record:

- Image-task training loss.
- Tabular-task training loss.
- Combined training loss.
- Image-task validation AUROC.
- Tabular-task validation AUROC.
- Image-task shared-gradient norm.
- Tabular-task shared-gradient norm.
- Cosine similarity between the task gradients in the shared layers.
- Learning rate.
- Training epoch.
- Selected checkpoint.

Negative gradient cosine similarity means that the tasks are requesting conflicting changes to the shared parameters.

The initial joint model will use equal task weights.

If persistent gradient conflict or strong task imbalance appears on permitted validation data, a documented optimization adjustment may be evaluated.

Permitted adjustments include:

- Gradient clipping.
- Revised fixed task weights.
- A learning-rate adjustment.
- PCGrad or another explicitly documented gradient-conflict method.

These adjustments remain variants of the same joint model.

No adjustment may be selected using either locked test split.

## 8. Evaluation protocol

### 8.1 Primary metric

The primary metric for both tasks is:

- ROC-AUC.

### 8.2 Secondary metrics

Secondary metrics are:

- Average precision or PR-AUC.
- Balanced accuracy.
- Accuracy.
- F1 score.
- Precision.
- Sensitivity.
- Specificity.
- Brier score.
- Expected calibration error.
- Calibration intercept.
- Calibration slope.
- Confusion matrix.

Accuracy will not be used as the only success measure.

No predetermined accuracy or AUROC will be claimed before evaluation.

### 8.3 Decision thresholds

A probability threshold of 0.5 will always be reported.

If a second operating threshold is used, it must be selected using permitted validation data only and frozen before locked-test evaluation.

No threshold may be selected separately after observing test performance.

### 8.4 Repeated training

The same joint architecture will be trained using at least three random seeds:

- 42.
- 123.
- 2026.

These repeated runs measure the statistical stability of the joint model.

Results will be reported as mean and standard deviation across seeds.

Where practical, 95% bootstrap confidence intervals will be calculated:

- By patient for SEER.
- By patient or slide for Camelyon17.
- By patient for the paired dataset.

## 9. Joint checkpoint-selection rule

One joint checkpoint will be selected using both permitted validation domains.

The checkpoint score will be:

Joint validation score =
0.5 × Camelyon17 OOD-validation AUROC
+
0.5 × SEER temporal-validation AUROC

Both task-specific AUROCs will always be reported separately so that strong performance on one task cannot hide poor performance on the other.

The SEER ID-validation results will be reported as an internal-distribution diagnostic but will not replace temporal-validation performance in the joint checkpoint score.

If checkpoints have joint validation scores within 0.001, the earlier checkpoint will be selected to reduce overfitting.

## 10. Test-set protection

### Camelyon17

- OOD-validation hospital 1 may be used for joint-model selection.
- OOD-test hospital 2 remains locked until the final joint-model configuration is selected.

### SEER

- Temporal validation from 2020–2021 may be used for joint-model selection.
- The 2022–2023 temporal test split remains locked until the final joint-model configuration is selected.

### Paired dataset

- Splits must be patient-level.
- No slide, patch or clinical record belonging to a test patient may appear during training.

No hyperparameter, feature, checkpoint, calibration method or threshold will be selected using a test set.

Each locked test set will be evaluated once after the model configuration is frozen.

## 11. Paired transfer and fusion

The paired-data phase is provisional until BCNB is downloaded and audited.

For each paired patient:

### Image branch

1. Divide the whole-slide image into tissue patches.
2. Encode each patch using the transferred DenseNet121 encoder.
3. Combine patch features using attention-based multiple-instance learning.
4. Produce one patient-level image vector.

### Clinical branch

1. Select clinical features available in both SEER and the paired dataset.
2. Apply preprocessing compatible with the transferred tabular encoder.
3. Apply the transferred tabular encoder.
4. Produce one patient-level clinical vector.

### Fusion

1. Pass the image and clinical vectors through compatible learned projections.
2. Concatenate the patient-level image and clinical vectors.
3. Apply a fusion multilayer perceptron.
4. Produce an 8-dimensional bottleneck.
5. Predict N0 versus N+.

Initial paired training will freeze the transferred encoders.

Later fine-tuning may unfreeze:

- The final DenseNet block.
- Modality projectors.
- Shared layers.

Small learning rates will be used during unfreezing.

All paired-data decisions remain provisional until the BCNB files, labels, clinical variables, patient identifiers and split possibilities have been audited.

## 12. Paired stage

The paired stage will use the image encoder, tabular encoder and shared representation learned by the joint model.

The paired stage will report:

- Patient-level image performance.
- Patient-level clinical performance where available.
- Patient-level fused performance.
- Calibration.
- The final 8-dimensional representation.

The paired model must use patient-level splits and must prevent slide or patch leakage across patients.

Any change required by the actual BCNB schema must be documented before paired-model training.

## 13. Classical and quantum handoff

The final paired fusion stage will export:

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

The quantum comparison is a downstream project stage and does not alter the training rules of the joint Camelyon17-SEER model.

## 14. Distinctive contribution

The project does not claim that multi-task learning or quantum classification is new by itself.

The distinctive system combines:

- Two large, unpaired cancer datasets.
- Two different data modalities.
- One joint neural network with two capabilities.
- A shared representation centered on lymph-node metastasis.
- Hospital distribution shift in histopathology.
- Temporal distribution shift in registry data.
- Explicit monitoring of task-gradient interaction.
- Transfer into a paired image-and-clinical dataset.
- A controlled classical-versus-quantum comparison.

The contribution will be demonstrated through transparent implementation, domain-shift evaluation, reproducible auditing and controlled downstream comparison.

## 15. Reporting rules

The project will:

- Separate exploratory pilot results from final results.
- Report both task-specific results and the joint checkpoint-selection score.
- Report unsuccessful training runs or optimization changes.
- Report the exact dataset split used for every result.
- Report metrics for each task separately.
- Clearly distinguish patch-level, tumor-record-level and patient-level predictions.
- Avoid statements such as “cancer-free” when the model evaluates only a patch or lymph-node status.
- Avoid presenting the system as a clinical diagnostic tool.
- Avoid claiming a predetermined performance level.
- Document all protocol amendments before the affected experiment is trained.
- Never expose SEER Patient IDs or patient-level source records in Git-tracked artifacts.

## 16. Execution order

1. Freeze the study protocol.
2. Complete reproducible SEER EDA and cohort construction.
3. Generate and freeze SEER patient-level splits.
4. Build and audit the leakage-safe SEER model table.
5. Complete development-only SEER drift analysis.
6. Implement train-only SEER preprocessing and the tabular neural encoder.
7. Implement the DenseNet121 image path and shared joint architecture.
8. Train the equal-weight joint model using alternating image and tabular mini-batches.
9. Monitor both task metrics, gradient norms and gradient cosine similarity.
10. Tune the joint model using only Camelyon17 OOD validation and SEER temporal validation.
11. Freeze the final joint-model configuration.
12. Download and audit the paired image-and-clinical dataset.
13. Transfer the learned encoders and shared representation to the paired dataset.
14. Train the paired fusion stage and export the fixed 8-dimensional representation.
15. Perform the controlled classical-versus-quantum comparison.
16. Evaluate each locked test set once.
17. Prepare the final report and presentation.
