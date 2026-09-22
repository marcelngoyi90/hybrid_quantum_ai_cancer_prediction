from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

from src.evidence_retrieval import retrieve_evidence
from src.llm_explanation import generate_background


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Dual Pathway Analysis")


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


@app.post("/api/predict")
async def predict(
    clinical: str = Form(...),
    image: UploadFile = File(...),
):
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

@app.get("/api/background")
async def background():
    knowledge_base = (
        PROJECT_ROOT / "knowledge_base" / "breast_node_evidence.json"
    )

    documents = retrieve_evidence(
        query=(
            "Camelyon lymph-node image patches, regional-node status, "
            "breast cancer staging, paired image and clinical research"
        ),
        knowledge_base_path=knowledge_base,
        top_k=4,
    )

    try:
        return await run_in_threadpool(generate_background, documents)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Background generation is temporarily unavailable.",
        ) from exc

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)