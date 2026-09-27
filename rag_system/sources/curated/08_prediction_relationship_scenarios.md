# Understanding the relationship between the two predictions

## The most important distinction

The Camelyon image pathway and the SEER clinical pathway do not analyse the
same evidence and, in this project, do not represent matched information from
the same patient.

- The image pathway examines one 96 by 96 lymph-node histopathology patch. Its
  label concerns tumour tissue in the central 32 by 32 region.
- The clinical pathway uses demographic and tumour characteristics from an
  already-diagnosed breast-cancer case. It predicts registry-defined regional
  lymph-node positivity, N0 versus N+.

The pathways share learned neural layers, but their two scores must not be
combined into one patient probability. A result from one pathway cannot confirm
or invalidate the other pathway.

## Image positive and clinical N+

An image-positive result means the model detected a tumour pattern in the
central region of the supplied lymph-node patch. An N+ clinical prediction
means the clinical variables were associated with one or more positive
regional nodes in the training cohort.

These outputs are conceptually consistent with regional lymph-node involvement,
but in this unpaired demonstration they are not two independent confirmations
about one real patient. Pathological staging requires complete clinical and
pathology assessment, not two research-model scores.

## Image positive and clinical N0

This combination is possible without a logical contradiction:

- The image model may detect a local tumour pattern in one patch.
- The clinical model makes a probabilistic prediction from indirect registry
  variables and can classify the case below its N+ threshold.
- The clinical model can make false-negative predictions.
- Threshold choice converts a continuous score into N0 or N+.
- The inputs are not a verified matched patient pair in this project.

For a real patient, tumour cells confirmed in regional lymph-node tissue are
important pathological evidence. The clinical model must never overrule a
pathologist or be presented as an official nodal stage.

## Image negative and clinical N+

This combination is also possible:

- One negative patch covers only a small central region.
- Tumour may be located elsewhere in the node, another node, or another patch.
- Sampling and model error can produce false-negative image results.
- The clinical pathway may identify a pattern associated with positive nodes.
- The inputs are unpaired, so the two outputs are not observations of the same
  patient.

A negative patch therefore does not mean that a patient has no breast cancer,
no nodal involvement, or no tumour anywhere else.

## Both outputs negative

The image model did not detect a tumour pattern in the labelled patch region,
and the clinical score fell below the N+ threshold. This still is not a medical
clearance: both outputs are research predictions, one patch is limited, and the
clinical target concerns patients already diagnosed with breast cancer.

## Why tumour and lymph nodes are related

Breast-cancer cells can travel through lymphatic vessels to nearby regional
lymph nodes. Sentinel nodes are among the first nodes likely to receive spread
from a primary tumour. Pathologists can examine removed nodes microscopically
for tumour cells, and nodal findings contribute to the N component of TNM
staging.

Regional lymph-node spread is not identical to distant metastatic breast
cancer. Regional nodes describe nearby spread; the M component describes
distant spread. The project uses the word metastasis in the Camelyon patch
label but must explain whether it refers to tumour in lymph-node tissue or
distant metastatic disease.

## Safe explanation template

An appropriate explanation should contain four parts:

1. State each result separately and identify its prediction unit.
2. Explain the biological relationship between breast tumours and regional
   lymph nodes.
3. Explain why agreement or disagreement can occur without combining scores.
4. State limitations and direct the user to qualified clinical assessment for
   real medical questions.

## Primary sources

- NCI breast-cancer diagnosis: https://www.cancer.gov/types/breast/diagnosis
- NCI TNM staging: https://www.cancer.gov/types/breast/stages/tnm-staging-system
- NCI sentinel-node biopsy: https://www.cancer.gov/about-cancer/diagnosis-staging/staging/sentinel-node-biopsy-fact-sheet
- WILDS Camelyon17 definition: https://wilds.stanford.edu/datasets/
- Project study protocol: reports/study_protocol.md

