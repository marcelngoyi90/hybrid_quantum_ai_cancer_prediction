from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from scripts.predict_camelyon_patch import THRESHOLD as IMAGE_THRESHOLD
from scripts.predict_dual_pathway import (
    CHECKPOINT_PATH,
    FEATURE_CONFIG_PATH,
    HISTORY_PATH,
    JOINT_CONFIG_PATH,
    PREPROCESSOR_STATE_PATH,
    SEED,
)
from scripts.predict_seer_patient import (
    THRESHOLD as CLINICAL_THRESHOLD,
    derive_breast_subtype,
    preprocess_record,
)
from src.camelyon17_data import build_transform
from src.joint_feature_extraction import load_selected_feature_context

from fastapi.concurrency import run_in_threadpool

from rag_system.llm import answer_question
from rag_system.retrieval import database_stats, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Dual Pathway Analysis")


class KnowledgeQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    level: str = Field(
        default="student",
        pattern="^(beginner|student|technical)$",
    )


class PathwayResult(BaseModel):
    predicted_class: str = Field(min_length=1, max_length=80)
    probability: float = Field(ge=0.0, le=1.0)


class PredictionExplanationRequest(BaseModel):
    image: PathwayResult
    clinical: PathwayResult | None = None
    level: Literal["beginner", "student", "technical"] = "student"


PREDICTION_FILES = {
    "Joint-model configuration": JOINT_CONFIG_PATH,
    "SEER feature configuration": FEATURE_CONFIG_PATH,
    "SEER preprocessor state": PREPROCESSOR_STATE_PATH,
    "Trained joint-model checkpoint": CHECKPOINT_PATH,
    "Training history": HISTORY_PATH,
}


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def prediction_readiness() -> dict:
    files = [
        {
            "name": name,
            "path": project_path(path),
            "available": path.is_file(),
        }
        for name, path in PREDICTION_FILES.items()
    ]
    missing = [item for item in files if not item["available"]]
    return {
        "ready": not missing,
        "model_seed": SEED,
        "files": files,
        "missing": missing,
        "image_input": {
            "width": 96,
            "height": 96,
            "formats": ["PNG", "JPEG"],
        },
        "message": (
            "The trained research model is ready for local analysis."
            if not missing
            else (
                "The interface and knowledge assistant are ready, but genuine "
                "prediction is disabled until the missing private model files "
                "are supplied."
            )
        ),
    }


@lru_cache(maxsize=1)
def model_context():
    return load_selected_feature_context(
        joint_config_path=JOINT_CONFIG_PATH,
        feature_config_path=FEATURE_CONFIG_PATH,
        preprocessor_state_path=PREPROCESSOR_STATE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        history_path=HISTORY_PATH,
        seed=SEED,
        requested_device="cpu",
    )


def checked_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be a whole number from {minimum} to {maximum}."
        )
    return value


def prepare_clinical_record(data, preprocessor):
    age = checked_integer(data.get("age"), "Age", 18, 90)
    size = checked_integer(data.get("tumour_size"), "Tumour size", 0, 400)

    fields = {
        "Race recode (W, B, AI, API)": "race",
        "Origin recode NHIA (Hispanic, Non-Hisp)": "origin",
        "Primary Site - labeled": "primary_site",
        "Histologic Type ICD-O-3": "histology",
        "Laterality": "laterality",
        "grade_unified": "grade",
        "ER Status Recode Breast Cancer (1990+)": "er",
        "PR Status Recode Breast Cancer (1990+)": "pr",
        "Derived HER2 Recode (2010+)": "her2",
        "t_stage_harmonized": "t_stage",
        "m_stage_harmonized": "m_stage",
    }

    record = {
        "age_numeric": age,
        "tumor_size_mm": size,
        "tumor_size_status": "measured",
    }

    for feature, input_name in fields.items():
        value = data.get(input_name)
        vocabulary = preprocessor.category_vocabularies[feature]

        if input_name in {"race", "origin", "primary_site", "laterality", "histology", "t_stage", "m_stage"} and value is None:
            record[feature] = None
            continue

        if not isinstance(value, str) or value not in vocabulary:
            raise ValueError(f"Select a valid value for {input_name}.")

        record[feature] = value

    subtype = derive_breast_subtype(
        record["ER Status Recode Breast Cancer (1990+)"],
        record["PR Status Recode Breast Cancer (1990+)"],
        record["Derived HER2 Recode (2010+)"],
    )

    if subtype is None:
        subtype = data.get("breast_subtype")
        if not isinstance(subtype, str):
            raise ValueError(
                "Select a breast subtype when receptor status is unknown."
            )

    subtype_feature = "Breast Subtype (2010+)"
    if subtype not in preprocessor.category_vocabularies[subtype_feature]:
        raise ValueError("Select a valid breast subtype.")

    record[subtype_feature] = subtype
    return record


def prepare_image(contents):
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.size != (96, 96):
                raise ValueError(
                    "The image must be a 96 × 96 lymph-node patch."
                )
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    return build_transform(training=False)(image).unsqueeze(0)


@app.post("/api/predict-image")
async def predict_image(image: UploadFile = File(...)):
    if not prediction_readiness()["ready"]:
        raise HTTPException(503, "Prediction is not ready. Model files are missing.")
    if image.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(422, "Upload a PNG or JPEG image.")
    contents = await image.read(4_000_001)
    if len(contents) > 4_000_000:
        raise HTTPException(413, "The image file is too large.")
    try:
        tensor = prepare_image(contents)
        context = model_context()
        with torch.inference_mode():
            logit = context.model.forward_image(tensor.to(context.device)).reshape(-1)[0]
            score = float(torch.sigmoid(logit).item())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(503, "The trained image model could not be loaded.") from exc
    return {
        "model_seed": SEED,
        "image_pathway": {
            "tumour_pattern_probability": round(score, 6),
            "threshold": IMAGE_THRESHOLD,
            "predicted_class": "tumour_pattern_detected" if score >= IMAGE_THRESHOLD else "no_tumour_pattern_detected",
        },
    }


@app.post("/api/predict")
async def predict(
    clinical: str = Form(...),
    image: UploadFile = File(...),
):
    readiness = prediction_readiness()
    if not readiness["ready"]:
        names = ", ".join(item["name"] for item in readiness["missing"])
        raise HTTPException(
            status_code=503,
            detail=f"Prediction is not ready. Missing: {names}.",
        )

    if image.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(422, "Upload a PNG or JPEG image.")

    contents = await image.read(4_000_001)
    if len(contents) > 4_000_000:
        raise HTTPException(413, "The image file is too large.")

    try:
        data = json.loads(clinical)
        if not isinstance(data, dict):
            raise ValueError("Clinical input must be an object.")

        context = model_context()
        record = prepare_clinical_record(data, context.preprocessor)
        image_tensor = prepare_image(contents)

        numeric_tensor, categorical_tensor = preprocess_record(
            record,
            context.preprocessor,
            context.device,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model could not be loaded. Check the model "
                "readiness panel and private artifact files."
            ),
        ) from exc

    with torch.inference_mode():
        image_logit = context.model.forward_image(
            image_tensor.to(context.device)
        ).reshape(-1)[0]

        clinical_logit = context.model.forward_tabular(
            numeric_tensor,
            categorical_tensor,
        ).reshape(-1)[0]

        image_score = float(torch.sigmoid(image_logit).item())
        clinical_score = float(torch.sigmoid(clinical_logit).item())

    return {
        "model_seed": SEED,
        "input_warning": (
            "Omitted registry fields are encoded as missing. "
            "Performance of this simplified form has not been evaluated."
            if any(data.get(name) is None for name in ("race", "origin", "primary_site", "laterality", "histology", "t_stage", "m_stage"))
            else ""
        ),
        "image_pathway": {
            "tumour_pattern_probability": round(image_score, 6),
            "threshold": IMAGE_THRESHOLD,
            "predicted_class": (
                "tumour_pattern_detected"
                if image_score >= IMAGE_THRESHOLD
                else "no_tumour_pattern_detected"
            ),
        },
        "clinical_pathway": {
            "n_positive_probability": round(clinical_score, 6),
            "threshold": CLINICAL_THRESHOLD,
            "predicted_class": (
                "N+" if clinical_score >= CLINICAL_THRESHOLD else "N0"
            ),
        },
    }


@app.get("/api/model/status")
async def model_status():
    return prediction_readiness()


@app.post("/api/explain-results")
async def explain_results(request: PredictionExplanationRequest):
    allowed_image = {
        "tumour_pattern_detected",
        "no_tumour_pattern_detected",
    }
    if request.image.predicted_class not in allowed_image:
        raise HTTPException(422, "The image result class is not valid.")
    if request.clinical is not None and request.clinical.predicted_class not in {"N0", "N+"}:
        raise HTTPException(422, "The clinical result class is not valid.")

    question = (
        "Explain these two research-model outputs separately in clear language. "
        f"The Camelyon image pathway predicted "
        f"{request.image.predicted_class.replace('_', ' ')} with a score of "
        f"{request.image.probability:.4f}. The SEER clinical pathway predicted "
        f"{request.clinical.predicted_class if request.clinical else 'not evaluated'} with an N+ score of "
        f"{request.clinical.probability if request.clinical else 'not available'}. Explain what each evaluated output means, "
        "why they may agree or differ, and their limitations. State clearly "
        "that Camelyon and SEER are unpaired datasets, these are not results "
        "from one matched patient, the scores must not be combined, and this "
        "research system is not a medical diagnosis."
    )
    if request.clinical is None:
        question = (
            "Explain this image-only research result in clear language: "
            f"{request.image.predicted_class.replace('_', ' ')} with a model score of {request.image.probability * 100:.2f}%. "
            "It describes the central region of one 96 by 96 lymph-node histopathology patch. "
            "No clinical pathway was evaluated. Do not invent clinical findings. "
            "Explain the prediction without treating it as confirmed tumour. "
            "A high model score can still be wrong and is not measured accuracy. "
            "Keep the explanation to this image; a negative patch does not establish a cancer-free patient."
        )
    try:
        return await run_in_threadpool(
            answer_question,
            question,
            level=request.level,
            concise_result=request.clinical is None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, "The knowledge database is not ready.") from exc
    except RuntimeError as exc:
        detail = (
            "The Groq API key is not configured on this computer."
            if "GROQ_API_KEY" in str(exc)
            else "The result explanation is temporarily unavailable."
        )
        raise HTTPException(503, detail) from exc
    except Exception as exc:
        raise HTTPException(
            502,
            "The result explanation could not be generated.",
        ) from exc

@app.get("/api/background")
async def background():
    try:
        evidence = await run_in_threadpool(
            retrieve,
            (
                "Camelyon17 image patch central 32 SEER regional node N0 N+ "
                "unpaired datasets separate outputs"
            ),
            top_k=5,
        )
        return {
            "text": (
                "The Camelyon17 pathway examines a small lymph-node image "
                "patch and predicts whether metastatic tumour is present in "
                "its labelled central region. The SEER pathway uses tabular "
                "registry information from people already diagnosed with "
                "breast cancer and predicts regional lymph-node status as N0 "
                "or N+. The tasks are medically related through lymph-node "
                "involvement, but they use different evidence. [S1, S2]\n\n"
                "The datasets are unpaired: their rows are not the same "
                "patients. Therefore, the two outputs must be reported "
                "separately and cannot be combined into one patient risk. A "
                "negative image patch only describes that sampled region, and "
                "neither pathway provides a complete diagnosis or cancer "
                "stage. [S3, S4, S5]"
            ),
            "sources": [
                {
                    "id": f"S{index}",
                    "title": item["title"],
                    "url": item["source_url"],
                }
                for index, item in enumerate(evidence, start=1)
            ],
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The knowledge database is not ready.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Background generation is temporarily unavailable.",
        ) from exc


@app.get("/api/knowledge/status")
async def knowledge_status():
    return database_stats()


@app.post("/api/ask")
async def ask_knowledge_base(request: KnowledgeQuestion):
    try:
        return await run_in_threadpool(
            answer_question,
            request.question,
            level=request.level,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge database is not ready. Run "
                "python -m rag_system.build_database."
            ),
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        detail = (
            "The Groq API key is not configured on this computer."
            if "GROQ_API_KEY" in message
            else "The teaching assistant is temporarily unavailable."
        )
        raise HTTPException(status_code=503, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The teaching assistant could not generate an answer.",
        ) from exc

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)
