# RAG knowledge system

This directory contains the evidence-grounded retrieval-augmented generation
(RAG) system for the project. It can teach a beginner, support an oral defence,
and explain project predictions and results using cited evidence.

## What it contains

- `sources/curated/`: original, RAG-ready knowledge articles synthesised from
  authoritative external sources and verified project files.
- `sources/source_catalog.csv`: traceable catalogue of external and internal
  evidence.
- `build_database.py`: creates the local SQLite knowledge database.
- `retrieval.py`: retrieves relevant chunks using word and character TF-IDF.
- `llm.py`: sends only the retrieved evidence and question to Groq.
- `prompts.py`: citation, medical-safety and explanation-level rules.
- `evaluations/questions.json`: retrieval questions covering all major topics.
- `tests/`: database and retrieval unit tests.

## Knowledge coverage

The current build indexes 30 documents into 277 searchable chunks. Coverage
includes:

- original CAMELYON17 whole slides and the Camelyon17-WILDS patch task;
- SEER registry variables, EOD coding, regional nodes and subtype recodes;
- NCI breast-cancer staging, TNM, grade, ER, PR and HER2;
- pathology reports, sentinel nodes and lymph-node tumour burden;
- project architecture, training, metrics, locked results and limitations;
- an oral-examination guide with common project questions.

The curated articles do not replace the primary sources. They provide readable
explanations and point back to the original evidence.

## Build the database

From the repository root, with the virtual environment active:

```powershell
python -m rag_system.build_database
```

The generated file is `rag_system/database/knowledge.db`. It is deliberately
ignored by Git because it is reproducible from the reviewed source documents.

## Configure Groq safely

Copy `.env.example` to `.env`, then place the key only in the local `.env` file:

```text
GROQ_API_KEY=your_private_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

`.env` is ignored by Git. Never paste the key into code, screenshots, reports,
issues, commits or chat messages.

## Start the website

```powershell
python -m uvicorn frontend.server:app --reload
```

Open `http://127.0.0.1:8000` and use **Knowledge assistant**. The question box
is open-ended, not a fixed FAQ: users can ask any question related to the
project and its wider medical, data-science, AI, quantum-computing, statistical
or software subjects. The interface answers in English and supports beginner,
student or technical explanations.

Retrieved evidence is used first. If no matching local source exists, the LLM
may give a clearly labelled general explanation. General explanations are not
presented as project facts and specialised or current claims require additional
verification.

The website checks prediction readiness automatically. Genuine local prediction
needs these files:

```text
configs/joint_model_v2.json
configs/seer_features_v2.json
artifacts/features/seer_v2_preprocessor.json
artifacts/checkpoints/joint_v2/joint_seed123_best.pt
artifacts/metrics/joint_v2/joint_seed123_history.json
```

If any file is absent, the page lists it and disables **Run analysis** instead
of returning a broken or invented result. The knowledge assistant remains
available and does not require private patient data.

After a genuine prediction, **Explain results** sends only the two output
classes and probabilities to the evidence-grounded assistant. It does not send
the entered clinical fields or image. The explanation keeps the image and
clinical pathways separate because the datasets are unpaired.

## Verify locally

```powershell
python -m unittest discover -s rag_system/tests -v
python -m rag_system.evaluate_retrieval
```

The first command verifies database construction and representative retrieval.
The second checks 60 questions across the knowledge areas. LLM wording and citation quality must
also be reviewed manually because correct retrieval alone cannot guarantee a
perfect generated answer.

## How an answer is produced

1. The English question is used as the retrieval query.
2. The local database ranks relevant evidence chunks.
3. The selected chunks, question and safety rules are sent to Groq.
4. The answer includes `[S1]`, `[S2]` citations and the interface lists the
   corresponding source links.
5. When evidence is insufficient, the assistant must say so instead of guessing.

## Files that should also be indexed directly

- `../README.md`
- `../reports/study_protocol.md`
- `../reports/model_v2_development_selection.md`
- `../reports/joint_v2_8d_validation.md`
- `../reports/final_locked_feature_protocol.md`
- `../configs/joint_model_v2.json`
- `../joint_v2/README.md`
- `../joint_v2/experiment_config.json`
- `../joint_v2/results/development/three_seed_development_summary.json`
- `../joint_v2/results/locked_test/FINAL_LOCKED_TEST_REPORT.md`
- `../joint_v2/results/locked_test/final_locked_test_results.json`

## Prohibited content

Do not index raw SEER records, patient identifiers, private feature arrays,
passwords, API keys, encrypted handoffs, or private model checkpoints.
