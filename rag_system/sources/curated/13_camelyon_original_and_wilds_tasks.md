# Original CAMELYON17 and the Camelyon17-WILDS task

## Original challenge data

The original CAMELYON17 challenge contains hematoxylin-and-eosin-stained
lymph-node whole-slide images from five medical centres in the Netherlands.
Whole-slide images are multi-resolution files that are much larger than normal
photographs. Expert-supervised annotations include lesion information for a
subset, and the original challenge also concerns patient-level pathological
node staging.

The challenge description reports 100 training patients and 100 test patients,
with five slides per patient, for 1,000 slides. These original slides and
patient-level challenge tasks must not be confused with the patch-classification
unit used by WILDS and by this project.

## WILDS adaptation

Camelyon17-WILDS turns the data into a binary patch-classification benchmark.
An input is a 96 by 96 RGB patch. Its label is determined by whether tumour is
present in the central 32 by 32 region. Hospital identity is the domain
metadata used to construct distribution-shift splits.

The project's frozen protocol uses hospitals 0, 3 and 4 for training, an
in-distribution split from those training hospitals, hospital 1 for OOD
validation and hospital 2 for locked OOD testing. Exact counts and split rules
come from the project protocol and WILDS implementation.

## Why hospitals create distribution shift

Hospitals can differ in staining, scanners, preparation and populations. A
model can learn signals that work well in its training domains but transfer
less reliably to a new hospital. OOD validation and testing ask whether useful
discrimination survives that domain change.

This does not mean that hospital identity causes tumour. Hospital is a domain
indicator, not the prediction target.

## Image terminology

- Whole-slide image: a digitized microscope slide covering a large tissue area.
- Patch: a small crop extracted from a slide.
- H&E: hematoxylin-and-eosin staining used to visualize tissue structures.
- Central-region label: the WILDS patch target refers to the middle of a patch,
  not necessarily every pixel in its surrounding context.
- Lesion annotation: an expert-marked tumour region on a slide.

## Safe interpretation

A patch classifier is not the same as a whole-slide diagnosis or patient-level
pN stage classifier. The model should state its prediction unit every time it
explains a result.

## Primary sources

- CAMELYON17 data: https://camelyon17.grand-challenge.org/Data/
- WILDS datasets: https://wilds.stanford.edu/datasets/
- WILDS Camelyon17 implementation: https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py
- WILDS benchmark paper: https://arxiv.org/abs/2012.07421

