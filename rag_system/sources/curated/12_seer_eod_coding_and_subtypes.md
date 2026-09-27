# SEER EOD coding, registry variables and breast subtypes

## Registry data are coded observations

SEER data are population-based cancer-registry data. Variables follow defined
collection periods, coding manuals and recodes. A code is not automatically a
direct biological measurement: it may summarize clinical, pathological or
registry information according to rules that can change across diagnosis years.
This motivates the project's temporal validation.

## Extent of Disease 2018

The SEER Extent of Disease (EOD) system contains primary-tumour, regional-node
and metastatic-disease fields. Site-specific schemas and algorithms can derive
stage-related variables using site, histology, nodes, tumour size and other data
items. EOD regional nodes classify regional involvement at diagnosis and are
separate from EOD metastatic disease.

SEER documentation distinguishes clinical assessment from pathological
assessment in relevant breast regional-node codes. Assessment can involve
imaging, needle sampling, sentinel-node biopsy, excision or dissection under
the applicable rules. Unknown or unassessable information must not be silently
treated as a confirmed negative.

## Regional nodes for breast cancer

The applicable regional-node definition may include ipsilateral axillary,
internal mammary, intramammary, infraclavicular and supraclavicular groups.
Coding can depend on node location, how nodes were assessed, tumour-deposit
size and numbers of positive nodes.

The project's N0/N+ target deliberately simplifies this richer information to
a binary research task. It should be described as registry-defined regional
node status, not as a replacement for full AJCC or EOD staging.

## Breast subtype recode

SEER's Breast Subtype (2010+) uses hormone-receptor status and HER2 status.
Hormone receptor positive means ER or PR is positive. Combining HR and HER2
produces categories such as HR+/HER2+, HR+/HER2-, HR-/HER2+ and triple-negative
(HR-/HER2-), with unknown/borderline and unavailable cases handled by the
published algorithm.

The project derives the same broad subtype logic from coded ER, PR and HER2
inputs when possible. It preserves an explicit category when receptor values
do not permit a reliable derived subtype. A model association involving subtype
does not establish treatment response or biological causation.

## Temporal interpretation

HER2 collection in SEER began in 2010, while variable definitions and registry
practices can evolve. A model trained on earlier years may face changes in
population, measurement or coding in later years. The temporal split estimates
performance under this type of time-related distribution shift.

## Primary sources

- SEER EOD introduction: https://training.seer.cancer.gov/eod/introduction.html
- SEER breast regional nodes: https://training.seer.cancer.gov/breast/staging/eodn.html
- SEER breast schema: https://staging.seer.cancer.gov/eod_public/schema/3.4/breast/
- SEER Breast Subtype (2010+): https://seer.cancer.gov/seerstat/databases/ssf/breast-subtype.html
- SEER variable definitions: https://seer.cancer.gov/analysis/

