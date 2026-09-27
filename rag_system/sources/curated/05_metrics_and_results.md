# Evaluation metrics and project results

## Confusion-matrix terms

- True positive: positive case predicted positive.
- False positive: negative case predicted positive.
- True negative: negative case predicted negative.
- False negative: positive case predicted negative.

## Threshold-dependent metrics

Accuracy is the fraction of all predictions that are correct. It can be
misleading when classes are imbalanced.

Sensitivity or recall is `TP / (TP + FN)`: the fraction of actual positive
cases detected.

Specificity is `TN / (TN + FP)`: the fraction of actual negative cases rejected.

Precision is `TP / (TP + FP)`: the fraction of positive predictions that are
correct.

F1 is the harmonic mean of precision and recall. It is useful when positive
class retrieval matters but does not include specificity directly.

Balanced accuracy is the mean of sensitivity and specificity for binary tasks.
It prevents the majority class from dominating the reported accuracy.

All these values depend on the classification threshold; the project uses 0.5
for its frozen downstream comparison.

## Ranking and probability metrics

ROC-AUC evaluates ranking across thresholds. A value near 1 indicates that
positive examples usually receive higher scores than negative examples. It does
not describe one operating threshold and is not the same as accuracy.

Average precision summarises precision-recall behaviour and focuses on the
positive class. Its baseline depends on positive-class prevalence, so values
from datasets with different prevalence should not be compared casually.

Brier score is the mean squared difference between predicted probability and
the binary outcome. Lower is better. It reflects both calibration and
resolution and should not be described as a pure calibration measure without
qualification.

## Mean and sample standard deviation

The project reports mean plus or minus sample standard deviation across seeds
42, 123, and 2026. The mean summarises central performance. The standard
deviation shows variation across the three runs. Three seeds improve
transparency but do not provide a large-sample uncertainty analysis.

## Joint-model development results

The upstream multi-task model achieved approximately:

- Camelyon OOD-validation ROC-AUC: 0.9789 +/- 0.0029.
- Camelyon OOD-validation accuracy: 92.84% +/- 1.15%.
- SEER temporal-validation ROC-AUC: 0.7946 +/- 0.0016.
- SEER temporal-validation accuracy at threshold 0.5: 75.99% +/- 0.13%.

An internal-validation-selected SEER threshold near 0.311 changed the operating
trade-off toward higher sensitivity. It did not retrain the model or change
ROC-AUC. The later downstream comparison froze threshold 0.5.

## Downstream development results

| Task | Classical ROC-AUC | Quantum ROC-AUC | Quantum minus classical |
|---|---:|---:|---:|
| Camelyon OOD validation | 0.978806 +/- 0.002870 | 0.974687 +/- 0.002632 | -0.004118 +/- 0.002229 |
| SEER temporal validation | 0.794835 +/- 0.001341 | 0.792162 +/- 0.001102 | -0.002673 +/- 0.000615 |

## Final locked-test results

| Task | Classical ROC-AUC | Quantum ROC-AUC | Quantum minus classical |
|---|---:|---:|---:|
| Camelyon OOD test | 0.979976 +/- 0.001784 | 0.973123 +/- 0.006888 | -0.006853 +/- 0.005617 |
| SEER temporal test | 0.797122 +/- 0.001216 | 0.794584 +/- 0.000899 | -0.002537 +/- 0.000424 |

The primary conclusion is narrow: the hybrid quantum model did not outperform
the parameter-matched classical model on mean locked-test ROC-AUC for either
task. This experiment therefore does not demonstrate quantum advantage.

## What may explain the pattern

Documented facts:

- Camelyon and SEER are different tasks with different inputs and targets.
- Camelyon patches are class-balanced in the benchmark; SEER N+ prevalence is
  about 31%.
- Camelyon tests hospital shift; SEER tests temporal shift.
- The classical and quantum downstream models are extremely small and receive
  already informative 8D representations.
- The classical model has 31 parameters and the hybrid model has 29.

Reasonable but unproven interpretations:

- Eight PCA features may already be easy for a small classical model to rank.
- The particular four-qubit circuit may not offer a useful inductive bias for
  these representations.
- Simulation does not provide hardware speedup.
- SEER outcomes may depend on information absent from the selected registry
  fields.

These interpretations require further controlled experiments before being
stated causally.

## Primary sources

- scikit-learn metrics API: https://scikit-learn.org/stable/api/sklearn.metrics.html
- scikit-learn PCA API: https://scikit-learn.org/stable/api/sklearn.decomposition.html
- Project README: README.md
- Downstream README: joint_v2/README.md
- Development summary: joint_v2/results/development/three_seed_development_summary.json
- Locked results: joint_v2/results/locked_test/final_locked_test_results.json
- Locked report: joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md

