# Medical context for the project

## Breast cancer and lymph nodes

Cancer begins when abnormal cells grow uncontrollably. Breast cancer originates
in breast tissue. Cancer cells may enter lymphatic vessels and reach nearby
regional lymph nodes. Sentinel lymph nodes are the first nodes to which cancer
cells are likely to spread from a primary tumour, so examining them can help
stage disease.

Regional lymph-node involvement and distant metastatic breast cancer are not
identical concepts. Regional spread refers to nearby lymph nodes. Distant
metastasis means spread to distant organs or distant nodes and corresponds to
the M component of TNM staging.

## Pathology and histopathology

Histopathology is the microscopic examination of prepared tissue. In the
CAMELYON data, lymph-node sections are stained with hematoxylin and eosin (H&E).
Hematoxylin and eosin make cellular structures visible with characteristic
colours, but appearance can vary with laboratory processing and scanning.

A whole-slide image is a digital scan of a large microscope slide. Because it
can be enormous, machine-learning pipelines commonly process small regions or
patches. A positive patch identifies tumour in a local region; it is not by
itself a complete assessment of every slide, every node, or the patient.

## TNM staging

TNM summarises three different aspects:

- T describes the primary tumour.
- N describes regional lymph-node involvement.
- M describes distant metastasis.

N0, N1, N2, and N3 have detailed clinical and pathological definitions. The
project's SEER target is a simplified binary N0-versus-N+ grouping based on the
number/status variable, while Camelyon-WILDS predicts local tumour presence in
a patch. Their targets share a lymph-node-metastasis theme but are not the same
clinical outcome.

## Biomarkers used in SEER features

- ER is the estrogen receptor.
- PR is the progesterone receptor.
- HER2 is human epidermal growth factor receptor 2.
- Hormone-receptor and HER2 combinations define clinically meaningful breast
  cancer subtype groups in the SEER recode.

These biomarkers are predictive inputs in the project. They must not be
described as causes established by the model. A predictive association does not
prove that changing a feature would change the outcome.

## Safety boundary

This project evaluates research models on retrospective datasets. It does not
replace pathology review, staging procedures, medical imaging, biopsy, or
clinical judgement. Dataset labels can contain coding limitations, and good
discrimination does not automatically imply safe clinical decisions.

## Primary sources

- NCI breast-cancer diagnosis and sentinel nodes: https://www.cancer.gov/types/breast/diagnosis
- NCI TNM staging: https://www.cancer.gov/types/breast/stages/tnm-staging-system
- NCI metastatic breast cancer: https://www.cancer.gov/types/breast/stages/metastatic-breast-cancer
- NCI sentinel-node biopsy: https://www.cancer.gov/about-cancer/diagnosis-staging/staging/sentinel-node-biopsy-fact-sheet
- CAMELYON17 data description: https://camelyon17.grand-challenge.org/Data/
- SEER breast-node training: https://training.seer.cancer.gov/breast/staging/eodn.html

