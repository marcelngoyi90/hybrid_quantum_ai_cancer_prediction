# Limitations and safe interpretation

## Dataset relationship

Camelyon and SEER contain different people and different prediction units. The
shared clinical theme is lymph-node metastasis, but patch tumour detection and
registry-defined node positivity are not identical targets. The project is
multi-task representation learning, not patient-level multimodal fusion.

## Dataset limitations

- Camelyon labelled patches are deliberately near-balanced and do not express
  natural whole-slide tumour prevalence.
- Camelyon represents five hospital domains, not every possible hospital.
- SEER is an observational registry with coding rules, missingness, and changes
  over time.
- The selected SEER variables cannot capture every biological, diagnostic, or
  treatment factor associated with nodal status.
- A registry-derived N0/N+ target is simpler than complete clinical staging.

## Modelling limitations

- The shared representation can experience task-gradient conflict.
- PCA is linear and may discard low-variance information even when total
  retained variance is high.
- Only one compact classical and one compact quantum downstream architecture
  were compared.
- The quantum model ran on a simulator rather than physical hardware.
- Three seeds reveal some instability but do not provide exhaustive uncertainty
  quantification.

## Evaluation limitations

- The locked tests provide protected final evaluation but are still drawn from
  the benchmark/registry data sources used by the study.
- No additional independent external cohort was evaluated.
- Metrics do not directly measure patient benefit, clinical workflow, fairness,
  or treatment outcomes.
- A threshold changes sensitivity and specificity without changing ROC-AUC.
- Comparing raw average precision across datasets is complicated by different
  positive-class prevalence.

## Safe claims

Supported:

- Both models learned useful signal in the fixed 8D representations.
- The classical model had higher mean locked-test ROC-AUC on both tasks.
- The hybrid model was close in absolute ROC-AUC but did not outperform the
  classical comparator.
- The protocol protected locked tests from fitting and tuning.

Unsupported:

- The model diagnoses cancer in the general population.
- A negative Camelyon patch proves a patient is cancer-free.
- The SEER model identifies whether a person has breast cancer.
- The experiment proves classical machine learning is always superior.
- The experiment demonstrates quantum advantage or quantum speedup.
- Observed feature associations are causal.
- The prototype is ready for clinical deployment.

## How the RAG should handle “why” questions

When a source directly reports a reason, state it as documented. When the
evidence only supports possible explanations, use language such as “may,”
“could,” or “one plausible interpretation,” and list competing explanations.
When no evidence is available, say that the study did not establish the cause
and propose the additional experiment needed to test it.

