# Pathology and regional lymph-node assessment

## What pathology does

Pathology studies cells and tissues to identify disease. A pathologist examines
a biopsy or surgical specimen and writes a pathology report. A report normally
describes the specimen and its origin, microscopic findings and a final
diagnosis. Breast pathology may also report grade, invasive status, biomarkers
and regional-node findings.

This complete professional process is fundamentally different from this
project's image classifier, which receives one prepared 96 by 96 patch and
predicts the label of its central region.

## Sentinel lymph nodes

Lymphatic drainage can carry breast-cancer cells to nearby nodes. A sentinel
node is among the first nodes likely to receive spread from a primary tumour.
In a sentinel lymph-node biopsy, one or more sentinel nodes are located,
removed and examined for cancer cells. The clinical interpretation depends on
the full specimen, procedures and patient context.

## Regional versus distant nodes

Breast regional nodes can include ipsilateral axillary levels, internal
mammary, intramammary, infraclavicular and supraclavicular nodes under the
applicable coding/staging system. SEER EOD rules distinguish regional nodes
from distant nodes, which belong to metastatic-disease coding.

## Tumour burden in a lymph node

Pathological nodal categories can consider the size of tumour deposits and the
number and location of involved nodes. The College of American Pathologists
protocol distinguishes isolated tumour cells, micrometastases and larger
metastases for pathological classification. These detailed categories cannot
be reconstructed from the project's binary patch label or binary N0/N+ target.

The words tumour pattern or metastatic tumour in the Camelyon label describe
tumour cells in lymph-node tissue. They do not, by themselves, specify a full
pN category, deposit size, total positive-node count or distant M category.

## Why one patch can miss tumour

A whole-slide image is far larger than a model patch. The WILDS task labels the
central 32 by 32 region of a 96 by 96 patch. A negative central region cannot
exclude tumour elsewhere in the slide, node, other nodes or unsampled tissue.
Sampling, staining, scanning and model error also affect a prediction.

## Safe explanation rule

For a real patient, pathology and clinical staging take priority. The model may
demonstrate research concepts, but it must not announce a definitive diagnosis,
node category or treatment decision.

## Primary sources

- NCI pathology reports: https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/pathology-reports-fact-sheet
- NCI breast-cancer diagnosis and sentinel-node biopsy: https://www.cancer.gov/types/breast/diagnosis
- SEER breast EOD regional nodes: https://training.seer.cancer.gov/breast/staging/eodn.html
- CAP invasive-breast resection protocol: https://documents.cap.org/protocols/Breast.Invasive_4.10.0.0.REL.CAPCP_R.pdf

