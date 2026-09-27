# Camelyon17 and Camelyon17-WILDS knowledge

## Beginner explanation

Camelyon17 contains microscope images of lymph-node tissue from breast-cancer
patients. The original slides are extremely large, so the WILDS benchmark uses
small image patches. A model receives one patch and predicts whether the
labelled central area contains tumour tissue.

A negative patch does not prove that the patient is cancer-free. It only means
that the labelled central area of that patch does not contain tumour according
to the dataset label.

## Original challenge versus WILDS benchmark

The original CAMELYON17 challenge used H&E-stained whole-slide images of lymph
node sections and included data from five medical centres. The challenge
addressed lesion detection and patient-level lymph-node staging. The WILDS
adaptation is a patch-level binary-classification benchmark designed to study
generalisation across hospitals.

The two names therefore should not be treated as perfectly interchangeable:

- CAMELYON17 refers to the original whole-slide challenge and associated data.
- Camelyon17-WILDS refers to the patch-based domain-generalisation benchmark
  used by this project.

## Input and label

Each WILDS input is a 96 by 96 RGB histopathology patch. The binary target is
one when the central 32 by 32 region contains any tumour tissue and zero
otherwise. The hospital is the domain variable. Official WILDS metadata also
tracks slide identity.

## Official project split

The repository audit records 455,954 labelled patches:

| Split | Patches | Centres | Non-tumour | Tumour | Tumour percentage |
|---|---:|---|---:|---:|---:|
| Train | 302,436 | 0, 3, 4 | 151,046 | 151,390 | 50.06% |
| ID validation | 33,560 | 0, 3, 4 | 16,952 | 16,608 | 49.49% |
| OOD validation | 34,904 | 1 | 17,452 | 17,452 | 50.00% |
| Locked OOD test | 85,054 | 2 | 42,527 | 42,527 | 50.00% |

The training split contains 26 patients and 30 slides; OOD validation contains
8 patients and 10 slides; locked OOD test contains 9 patients and 10 slides.
The project audit checks that OOD splits do not overlap other splits in patient,
slide, or image identifier.

## Why hospital shift matters

Histopathology appearance can vary across hospitals because of staining,
tissue preparation, scanners, acquisition procedures, and patient populations.
A model can perform well on familiar hospital images yet degrade on a new
hospital. Camelyon17-WILDS treats hospitals as domains so OOD evaluation measures
this form of real-world distribution shift.

The class-balanced labelled patches do not represent the natural prevalence of
tumour across every location in a whole slide. The WILDS documentation notes
that the underlying patch distribution is much more negative. Consequently,
classification performance on the balanced benchmark should not be interpreted
as a population prevalence estimate.

## Project preprocessing

All images are resized to 96 by 96. Training-only augmentation applies random
horizontal and vertical flips, rotations of 0, 90, 180, or 270 degrees, and
moderate colour jitter. Images are converted to tensors and normalised with
ImageNet channel means and standard deviations. Evaluation data is resized and
normalised without random augmentation.

Augmentation creates plausible variations and reduces memorisation. It does not
create new independent patients and cannot by itself guarantee robustness to
every hospital.

## Project image architecture

The selected image path is:

```text
96x96x3 image
-> ImageNet-initialised DenseNet121
-> 1024-dimensional encoder output
-> 1024-to-256-to-128 projector
-> shared 128-to-64-to-32 trunk
-> image-specific 32-to-1 logit head
```

All DenseNet layers are fine-tuned. DenseNet uses dense connections so later
layers can reuse earlier feature maps and gradients have shorter routes through
the network. ImageNet initialisation supplies general visual features, while
fine-tuning adapts them to histopathology.

## Evaluation interpretation

ID validation uses familiar training hospitals. OOD validation and OOD test use
unseen hospitals, so they are more relevant to hospital generalisation. The OOD
validation split supported development and checkpoint selection; the OOD test
remained locked until the downstream protocol was frozen.

The downstream final locked-test comparison produced:

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Classical neural | 0.979976 | 0.982969 | 0.918440 | 0.918440 | 0.913276 |
| Hybrid quantum | 0.973123 | 0.977738 | 0.919075 | 0.919075 | 0.914269 |

The classical model had stronger ranking performance, the predeclared primary
comparison. At the fixed 0.5 threshold, the hybrid model had marginally higher
mean accuracy and F1. This is not contradictory: ROC-AUC evaluates ranking over
thresholds, while accuracy and F1 depend on the chosen threshold.

## Limitations

- Patch-level prediction is not patient-level staging.
- Balanced sampling changes the apparent class prevalence.
- Five hospital domains cannot represent every future deployment setting.
- OOD performance can vary across random seeds.
- Strong benchmark performance does not establish clinical utility.
- The system is not approved for diagnosis or treatment decisions.

## Primary sources

- WILDS dataset page: https://wilds.stanford.edu/datasets/
- Official WILDS implementation: https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py
- CAMELYON17 data page: https://camelyon17.grand-challenge.org/Data/
- CAMELYON17 challenge home and paper reference: https://camelyon17.grand-challenge.org/
- WILDS paper: https://arxiv.org/abs/2012.07421
- Project audit: artifacts/metrics/eda_split_summary.csv
- Project loader: src/camelyon17_data.py
- Frozen model configuration: configs/joint_model_v2.json
- Final downstream report: joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md

