# Beginner glossary

## Dataset

A collection of examples used for research. Camelyon17 contains image patches;
SEER contains registry records.

## Model

A mathematical system that learns patterns from training data and produces a
score or prediction for new input.

## Training

The process in which a model adjusts its parameters using labelled examples.

## Validation

Data used during development to compare checkpoints or choices. It is separate
from training data.

## Locked test

Data protected until all decisions are frozen. It is used for final evaluation,
not for improving the model after seeing its results.

## Image patch

A small square taken from a much larger microscope slide. A patch is not the
same as the whole slide, lymph node, or patient.

## Histopathology

The microscopic study of prepared tissue to identify disease-related patterns.

## Tumour

An abnormal mass of tissue. Tumours can be benign or malignant; this project's
inputs and labels are specific to breast-cancer and lymph-node research data.

## Lymph node

A small structure in the lymphatic system that helps filter lymph and supports
immune function. Cancer cells can spread to regional lymph nodes.

## Metastasis

Spread of cancer cells from their original site. Regional lymph-node
involvement and distant metastasis are related but distinct clinical concepts.

## N0 and N+

In this project, N0 means no positive regional lymph nodes were recorded under
the target rule. N+ means one or more positive regional nodes, positive nodal
sampling, or documented positive nodes with unspecified count. This simplified
binary target is not the complete AJCC N-stage system.

## Feature

An input value used by a model, such as age, tumour size, receptor status, or a
number extracted from an image.

## Encoder

A model component that converts raw input into a useful numerical
representation.

## Shared representation

Features produced by neural layers used by more than one task. Shared does not
mean that Camelyon images are matched to SEER patients.

## Multi-task learning

Training one model on more than one related task while allowing some parameters
to be shared.

## PCA

Principal Component Analysis, a linear method that compresses features while
preserving directions with high variation. In this project it reduces 32
features to 8 using training data only.

## Qubit

The basic information unit of a quantum circuit. The project's qubits are
simulated on a classical computer.

## Hybrid quantum model

A model combining a parameterised quantum circuit with classical computation.

## Probability score

A number between zero and one produced after applying a sigmoid to a model
logit. It is a model score, not automatically a calibrated individual medical
risk.

## Threshold

A cutoff that turns a continuous score into a class. Changing the threshold can
change sensitivity and specificity without retraining the model.

## ROC-AUC

A measure of how well a model ranks positive examples above negative examples
across possible thresholds. It is not the same as accuracy.

## Sensitivity

The fraction of truly positive examples detected by the model.

## Specificity

The fraction of truly negative examples correctly rejected by the model.

## Precision

The fraction of positive predictions that are truly positive.

## RAG

Retrieval-Augmented Generation. A retrieval component first finds relevant
evidence; a language model then writes an answer grounded in that evidence.

## Citation

A reference showing which document supports a claim.

