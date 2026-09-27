# SEER breast-cancer registry knowledge

## Beginner explanation

SEER is a cancer-registry programme of the United States National Cancer
Institute. It collects population-based information about cancers, such as age,
tumour characteristics, stage, biomarkers, diagnosis year, and geographic
registry information.

In this project, every included person already has a breast-cancer diagnosis.
The model does not predict cancer versus no cancer. It predicts whether regional
lymph nodes are recorded as positive.

## Project target

The target is constructed from the SEER Regional Nodes Positive variable:

- Class 0, N0: code 00, meaning examined regional nodes are negative.
- Class 1, N+: codes 01-90, 95, or 97, meaning one or more positive nodes,
  positive sampling, or positive nodes with unspecified count.
- Unknown, invalid, no-node-exam, and unusable values are excluded according to
  the frozen cohort rules.

The binary project target is a pragmatic registry-derived grouping. It is not a
complete replacement for clinical AJCC N staging, which also considers the
location, number, and size of lymph-node metastases and how they were assessed.

## Cohort construction

The raw export contained 756,356 tumour records. The project restricted the
primary cohort to adults and selected one deterministic index breast-cancer
record per patient. The final cohort contains 731,278 unique patients:

- 503,794 N0 patients.
- 227,484 N+ patients.
- 31.11% N+ prevalence.

Selecting one record per patient avoids treating repeated tumours from the same
person as independent patients and helps prevent leakage between splits.

## Temporal splits

| Split | Diagnosis years | Patients | Purpose |
|---|---|---:|---|
| Train | 2010-2019 | 461,292 | Fit model and preprocessing |
| ID validation | 2010-2019 | 51,241 | Familiar-period validation |
| Temporal validation | 2020-2021 | 107,249 | Development selection under time shift |
| Locked temporal test | 2022-2023 | 111,496 | One-time final evaluation |

No patient occurs in more than one split. Diagnosis year is used for splitting
and aggregate auditing, not as a predictive model feature.

Temporal evaluation asks whether a model learned from earlier years continues
to work in later years. Changes in coding, screening, patient mix, treatment,
data completeness, and clinical practice may create temporal distribution
shift.

## Project features

Numeric features:

- Age.
- Tumour size in millimetres.

Categorical features:

- Race recode.
- Hispanic-origin recode.
- Primary breast site.
- Histologic type.
- Laterality.
- Unified tumour grade.
- Estrogen-receptor status.
- Progesterone-receptor status.
- HER2 status.
- Breast-cancer subtype.
- Tumour-size status.
- Harmonised T stage.
- Harmonised M stage.

Patient identifiers, year, split assignment, target-derived variables, regional
nodes positive, and regional nodes examined are not model inputs.

## Leakage-safe preprocessing

Numeric medians and scaling statistics are learned from training data only.
Categorical vocabularies are also learned from training data only. Missing or
unseen categories map to a reserved index rather than causing validation or test
information to alter the vocabulary.

The tabular encoder learns embeddings for categorical values, combines them
with numeric features, and produces a 256-dimensional representation followed
by a 128-dimensional projector. The shared trunk reduces this to 32 dimensions.

## Relevant medical context

The N category of TNM staging describes regional lymph-node involvement. N0
means no regional lymph-node metastasis is identified, with detailed staging
rules for isolated tumour cells. Higher N categories reflect patterns such as
the number and location of involved lymph nodes. The project intentionally uses
a simpler N0-versus-N+ target based on a specific registry variable.

ER, PR, and HER2 are biomarkers relevant to breast-cancer subtype. SEER began
collecting HER2 for breast cases in 2010, which is why the project period starts
in 2010 and can use the Breast Subtype (2010+) recode.

## Final downstream results

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Classical neural | 0.797122 | 0.653177 | 0.712052 | 0.718665 | 0.605214 |
| Hybrid quantum | 0.794584 | 0.642101 | 0.707873 | 0.716654 | 0.602893 |

The classical model was slightly stronger on the primary ROC-AUC and most
secondary metrics. The hybrid model had slightly higher mean sensitivity but
lower specificity. The results show predictive association in the curated
registry cohort; they do not establish causal effects or clinical usefulness.

## Why SEER performance is lower than Camelyon performance

The project documents the numerical difference but does not experimentally
prove a single cause. Reasonable, explicitly labelled interpretations include:

- The tasks and prediction units are different.
- Registry variables are indirect summaries rather than direct images of a
  labelled tumour region.
- Nodal status depends on biological and clinical factors not fully represented
  in the selected fields.
- Coding changes, missingness, and temporal shift add uncertainty.
- The N+ class is less common than N0.

The RAG assistant must not present these interpretations as proven causal
explanations.

## Privacy and use limitations

SEER data access is governed by data agreements. Raw records, patient-level
tables, identifiers, private feature archives, and train-fitted private states
are not part of the public RAG corpus. Only aggregate, privacy-safe project
information should be retrieved.

## Primary sources

- SEER Data & Software: https://seer.cancer.gov/data-software/
- SEER documentation: https://seer.cancer.gov/data-software/documentation/
- SEER coding manuals: https://seer.cancer.gov/tools/codingmanuals/
- SEER data access: https://seer.cancer.gov/data/access.html
- SEER data agreements: https://seer.cancer.gov/data/agreements.html
- Breast Subtype recode: https://seer.cancer.gov/seerstat/databases/ssf/breast-subtype.html
- SEER regional-node training: https://training.seer.cancer.gov/breast/staging/eodn.html
- NCI TNM staging: https://www.cancer.gov/types/breast/stages/tnm-staging-system
- Project protocol: reports/study_protocol.md
- Project configuration: configs/joint_model_v2.json
- Project audits: artifacts/metrics/seer_v2_cohort_audit.json and related files

