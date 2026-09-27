# Knowledge-corpus scope and answering rules

## Purpose

This curated corpus supports a retrieval-augmented generation (RAG) assistant
for the Hybrid Quantum AI for Cancer Prediction project. It is designed to
answer questions about the datasets, medical context, preprocessing, model
architecture, evaluation metrics, experiment results, limitations, and the
classical-versus-quantum comparison.

The corpus combines three evidence levels:

1. Project-primary evidence: frozen protocols, configurations, code, audit
   files, development summaries, and locked-test reports in this repository.
2. External-primary evidence: official dataset documentation, government
   cancer information, official software documentation, and original papers.
3. Curated explanation: plain-language synthesis that connects the primary
   evidence without changing its meaning.

## Required answer behaviour

The RAG assistant must:

- Answer from retrieved evidence rather than unsupported memory.
- Cite the document title and source URL for externally sourced claims.
- Cite repository paths for project-specific claims and results.
- Clearly separate documented facts from plausible interpretations.
- Say when available evidence is insufficient to identify a cause.
- Explain technical terms when the user is a beginner.
- Never expose patient identifiers, private arrays, API keys, passwords, or
  private checkpoints.
- Never describe the prototype as a clinical diagnostic system.
- Never equate a Camelyon class-0 patch with a cancer-free patient.
- Never say that SEER predicts whether a person has breast cancer; every member
  of the project cohort already has a breast-cancer diagnosis.
- Never claim quantum advantage because the frozen primary comparison did not
  demonstrate one.

## Supported explanation levels

- Beginner: short sentences, analogies, and definitions before metrics.
- Student: accurate project terminology with enough detail for an oral exam.
- Technical: exact splits, transformations, architectures, protocols, and
  quantitative results.

## Boundary of “all information”

No finite corpus can contain everything on the internet. In this project,
“all information” means the authoritative and project-relevant information
needed to explain what was studied, why it was studied, how it was implemented,
how it was evaluated, what the results mean, and what cannot be concluded.

