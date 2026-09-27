# Project oral-examination guide

## One-sentence project description

The project trains a hard-parameter-sharing multi-task network on unpaired
Camelyon17 image patches and SEER clinical records, then compares compact
classical and four-qubit hybrid quantum classifiers on frozen eight-dimensional
representations.

## Why use one joint model if the data are unpaired?

The research question is whether two related lymph-node tasks can contribute to
a useful shared representation. The data are never matched or fused by patient.
Each modality has its own encoder and output head; alternating mini-batches
update the same shared trunk.

## What makes this cancer AI project different or unique?

The contribution is not a claim that quantum models automatically improve
cancer prediction. It is a controlled, parameter-matched classical-versus-
hybrid-quantum comparison downstream of a jointly learned representation,
across two modalities, three seeds and protected locked tests.

## What does each task predict?

- Camelyon: tumour in the central region of a lymph-node image patch.
- SEER: binary registry-defined regional lymph-node status among diagnosed
  breast-cancer cases.

These outputs are scientifically related through nodal involvement but are not
two measurements from the same person.

## Why reduce 32 dimensions to 8?

The frozen shared representation is reduced using training-only
standardization and PCA to create a compact common input for both downstream
classifiers and a feasible input for the four-qubit circuit through feature
encoding. Validation and locked-test data do not fit those transformations.

## What made the comparison fair?

The classical and quantum downstream models used the same fixed 8D inputs,
splits, preprocessing, training objective, optimizer rules, batch sizes,
validation metric, threshold rules and seeds. They were also close in trainable
parameter count: 31 for the classical network and 29 for the hybrid model.

## What were the final results?

On locked tests, mean ROC-AUC was 0.979976 for classical versus 0.973123 for
hybrid quantum on Camelyon, and 0.797122 versus 0.794584 on SEER. The hybrid
model learned strong signal but did not outperform the classical comparator.
The study therefore did not demonstrate quantum advantage.

## Why is a negative result useful?

A well-controlled result showing no advantage prevents exaggerated claims. It
establishes a reproducible baseline, identifies limitations and suggests future
tests of alternative circuits, representations, hardware and external cohorts.

## What are the main limitations?

The datasets are unpaired, targets are simplified, the quantum circuit was
simulated, only one compact architecture family was compared, three seeds do
not describe every uncertainty, and no independent clinical deployment study
was performed.

## What should never be claimed?

Do not say the system diagnoses cancer, assigns a complete stage, recommends
treatment, combines the two pathway scores into one patient risk, proves a
causal relationship or demonstrates quantum advantage.

## Where do the answers come from?

Project-specific claims come from the frozen protocol, configurations, code,
development summary and locked-test report. Medical and dataset context comes
from traceable NCI, SEER, WILDS, CAMELYON and pathology sources. The RAG system
retrieves relevant evidence before the Groq-hosted language model writes an
English explanation with grouped citations.
