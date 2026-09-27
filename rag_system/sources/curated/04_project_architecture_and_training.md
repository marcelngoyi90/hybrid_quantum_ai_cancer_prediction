# Project architecture and training

## Research design

The project studies whether a single network can learn useful shared features
from two large but unpaired cancer datasets. Camelyon17 supplies image patches;
SEER supplies tabular clinical records. No image corresponds to a SEER row, so
the data is not fused at the patient level.

## Hard-parameter-sharing multi-task learning

Each task has a modality-specific encoder and output head. Between them sits a
shared neural trunk. One image mini-batch and one SEER mini-batch are processed
in a training cycle. Their mean binary losses are weighted equally:

```text
total loss = 0.5 * image loss + 0.5 * tabular loss
```

This is hard parameter sharing because the two tasks update the same shared
parameters. Multi-task learning may help when related training signals support
a useful representation, but it may also create gradient conflict. The project
therefore records gradient norms and cosine similarity rather than assuming
that sharing is always beneficial.

The image and tabular 128-dimensional vectors are never concatenated, averaged
or fused together. Each modality passes through the same shared trunk in its
own forward pass. The mini-batches alternate, so an image example is never
paired with a SEER row.

## Architecture

```text
Camelyon image -> DenseNet121 -> image projector -> 128D
                                                     \
                                                      shared 128->64->32
                                                     /
SEER fields -> embeddings/tabular encoder -> projector -> 128D

32D -> image head for patch tumour label
32D -> tabular head for N0 versus N+
```

The outputs are always interpreted separately. The network does not produce one
universal cancer probability.

## Frozen joint-model training

- Random seeds: 42, 123, 2026.
- Image batch size: 32.
- Tabular batch size: 49.
- Optimizer: AdamW.
- Image encoder learning rate: 0.00005.
- Other main learning rates: 0.0003.
- Maximum epochs: 15.
- Gradient clipping maximum norm: 5.0.
- Warmup followed by cosine learning-rate decay.
- Checkpoint score: equal mean of Camelyon OOD-validation ROC-AUC and SEER
  temporal-validation ROC-AUC.

## Representation extraction and PCA

The selected checkpoint produces a frozen 32-dimensional shared representation
for each task. Evaluation mode is enabled, parameters are frozen, and original
row order is preserved. A separate StandardScaler and PCA are fitted for each
task and seed using training representations only:

```text
32D shared features -> training-only standardisation -> training-only PCA -> 8D
```

Validation and locked-test data are transformed with the already fitted states;
they are never used to refit them. All six development PCA pipelines retained
more than 99.91% of variance. Linear probes showed 99.83% ROC-AUC retention for
Camelyon and 99.98% for SEER.

## Downstream controlled comparison

Both downstream models receive the same 8D rows and the deterministic transform
`2 * arctan(feature)`, which maps real values to approximately `[-pi, pi]`.

Classical model:

```text
8 inputs -> 3 tanh hidden units -> 1 logit
31 trainable parameters
```

Hybrid quantum model:

- Four simulated qubits.
- Two data-reuploading rounds.
- Four features encoded with Y rotations per round.
- A strongly entangling layer after each upload.
- Four Pauli-Z expectation values.
- A final classical 4-to-1 layer.
- 29 trainable parameters.

Both use Adam, learning rate 0.01, batch size 1024, maximum 15 epochs, patience
4, weighted binary cross-entropy, validation ROC-AUC selection, and threshold
0.5.

## Locked-test integrity

Locked data is not used for preprocessing, architecture selection, training,
tuning, calibration, threshold selection, or seed selection. The downstream
runner originally failed to persist selected model states, so the 12 states
were recreated before loading locked arrays. Each was accepted only if its best
epoch and development ROC-AUC matched the archived value within 1e-6. Hashes
were audited, then locked inference was performed without fitting.

## Primary sources

- Project protocol: reports/study_protocol.md
- Development selection: reports/model_v2_development_selection.md
- Configuration: configs/joint_model_v2.json
- Joint architecture: src/joint_model.py
- Joint training: src/joint_training.py
- Downstream documentation: joint_v2/README.md
- Downstream models: joint_v2/models.py
- Final report: joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md
- DenseNet paper: https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html
- Caruana multi-task-learning paper: https://doi.org/10.1023/A:1007379606734
- PCA documentation: https://scikit-learn.org/stable/api/sklearn.decomposition.html
