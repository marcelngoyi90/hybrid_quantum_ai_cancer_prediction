# Breast-cancer staging, grade and biomarkers

## What stage means

Cancer stage describes the extent of disease. For breast cancer, stage is not
determined by one model score or one variable. It incorporates the primary
tumour, nearby lymph nodes, distant spread, tumour grade and biomarker status.
The exact interpretation belongs to qualified clinical teams using complete
patient information.

## Clinical and pathological prognostic staging

Clinical prognostic staging is performed before surgery. It can use health
history, physical examination, imaging and biopsy information. Pathological
prognostic staging is available for patients whose first treatment includes
surgery; it adds laboratory findings from removed breast tissue and lymph
nodes. These stages should not be confused with a machine-learning prediction.

The project does not reproduce either full process. Camelyon supplies local
histopathology patch evidence, while the SEER pathway estimates a simplified
registry-derived N0-versus-N+ target from selected tabular variables.

## TNM

TNM separates three concepts:

- T describes the size and location or extent of the primary tumour.
- N describes regional lymph-node involvement, including relevant location and
  tumour burden in nodes.
- M describes distant spread to other parts of the body.

Regional lymph-node involvement is therefore not identical to M1 distant
metastatic disease. In the project, references to tumour in a Camelyon
lymph-node patch concern regional node tissue. A patch result cannot establish
a complete T, N, M or stage group.

## Grade is not stage

Tumour grade describes how abnormal tumour cells and tissue appear under a
microscope and is related to expected growth behaviour. Stage describes the
extent and spread of cancer. A pathologist assesses breast-tumour features such
as tissue organization, nuclear appearance and cell division to assign a
grade. Grade is an input feature in the SEER pathway, not a model conclusion.

## ER, PR and HER2

Breast-cancer tissue is commonly tested for estrogen receptor (ER),
progesterone receptor (PR) and HER2. ER and PR are hormone receptors. HER2 is a
protein involved in cell growth. Their status contributes to subtype,
prognostic staging and treatment planning in real clinical care.

The project uses registry-coded ER, PR and HER2 variables as predictive inputs.
It does not run biomarker laboratory tests, prescribe therapy or infer that an
association is causal.

## Why this matters for model explanations

An explanation can say that tumour size, grade, T stage, M stage and receptor
status provide indirect information associated with nodal status. It must not
say these variables alone determine a patient's true stage or treatment. The
SEER output is a probability of the project's binary regional-node target, not
an automatically assigned clinical stage.

## Primary sources

- NCI breast-cancer stages: https://www.cancer.gov/types/breast/stages
- NCI breast-cancer diagnosis: https://www.cancer.gov/types/breast/diagnosis
- NCI biomarker tests: https://www.cancer.gov/types/breast/diagnosis/breast-cancer-biomarker-tests
- NCI tumour grade: https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-grade

