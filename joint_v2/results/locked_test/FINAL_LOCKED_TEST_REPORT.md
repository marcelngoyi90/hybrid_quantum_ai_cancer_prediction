# Joint Model v2 Final Locked-Test Report

## Status

This report records the one-time downstream locked-test evaluation for the
classical neural and hybrid quantum classifiers. Camelyon `ood_test` and SEER
`temporal_test` were evaluated after the downstream architecture, input
transformation, training configuration, model-selection rule, threshold, and
three-seed aggregation procedure were frozen.

No training, fitting, tuning, calibration, threshold selection, architecture
change, seed selection, or preprocessing change was performed using locked-test
data.

## Downstream state recovery disclosure

The development runner originally saved aggregate metrics and histories but did
not persist the selected downstream model state files. After the locked handoff
arrived, the upstream owner instructed the contributor to complete the locked
test using the frozen quantum architecture.

Before any locked array was loaded, all 12 downstream states were recreated
using only the original development training and validation features, seeds,
architectures, optimizer settings, early-stopping rule, and model-selection
metric. A state was accepted only when its best epoch and development ROC-AUC
reproduced the previously recorded result within an absolute tolerance of
`1e-6`. All 12 states passed. Their SHA-256 hashes are recorded in
`frozen_state_recovery_audit.json`; the state files remain private and excluded
from Git.

## Frozen comparison

Both models received the same eight-dimensional features transformed by:

```text
angle = 2 * arctan(feature)
```

The classification threshold remained fixed at `0.5`.

- Classical neural model: 31 trainable parameters.
- Hybrid four-qubit model: 29 trainable parameters.

Results are mean +/- sample standard deviation across seeds 42, 123, and 2026.

## Camelyon OOD test

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical neural | 0.979976 +/- 0.001784 | 0.982969 +/- 0.001511 | 0.918440 +/- 0.005571 | 0.918440 +/- 0.005571 | 0.913276 +/- 0.006783 |
| Hybrid quantum | 0.973123 +/- 0.006888 | 0.977738 +/- 0.005096 | 0.919075 +/- 0.010576 | 0.919075 +/- 0.010576 | 0.914269 +/- 0.013010 |

Mean quantum-minus-classical ROC-AUC:

```text
-0.006853 +/- 0.005617
```

The hybrid model had marginally higher mean accuracy and F1 at the fixed
threshold, but lower ROC-AUC and average precision. The classical model provided
stronger ranking performance, which was the predeclared primary comparison.

## SEER temporal test

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical neural | 0.797122 +/- 0.001216 | 0.653177 +/- 0.001943 | 0.712052 +/- 0.002791 | 0.718665 +/- 0.000696 | 0.605214 +/- 0.000834 |
| Hybrid quantum | 0.794584 +/- 0.000899 | 0.642101 +/- 0.003616 | 0.707873 +/- 0.010166 | 0.716654 +/- 0.001185 | 0.602893 +/- 0.001385 |

Mean quantum-minus-classical ROC-AUC:

```text
-0.002537 +/- 0.000424
```

The classical model had higher mean ROC-AUC, average precision, accuracy,
balanced accuracy, and F1. The hybrid model had slightly higher mean sensitivity
but lower specificity at the fixed threshold.

## Final conclusion

Both compact models generalized to the locked splits and learned strong
predictive signal. The hybrid quantum model did not outperform the
parameter-matched classical neural model on the primary locked-test ROC-AUC for
either task. Therefore, this experiment does not demonstrate a quantum
advantage.

The project remains an educational research prototype. It does not establish
clinical utility and must not be used for diagnosis or treatment decisions.
