const imageInput = document.querySelector("#image-file");
const preview = document.querySelector("#image-preview");
const subtypeOutput = document.querySelector("#subtype");
const manualSubtypeField = document.querySelector("#manual-subtype-field");
const manualSubtype = document.querySelector("#manual-subtype");
const runButton = document.querySelector("#run-analysis");
const runStatus = document.querySelector("#run-status");

let imageReady = false;
let modelReady = false;
let latestPrediction = null;

function field(id) {
  return document.getElementById(id);
}

async function loadModelStatus() {
  const card = field("model-readiness");
  const title = field("model-readiness-title");
  const message = field("model-readiness-message");
  const missingList = field("model-missing-files");

  try {
    const response = await fetch("/api/model/status");
    const result = await response.json();
    if (!response.ok) throw new Error("Could not check the model files.");

    modelReady = result.ready;
    card.classList.add(modelReady ? "ready" : "unavailable");
    title.textContent = modelReady
      ? "Prediction model ready"
      : "Prediction model files are incomplete";
    message.textContent = result.message;
    runButton.disabled = !modelReady;

    missingList.replaceChildren();
    for (const item of result.missing) {
      const row = document.createElement("li");
      row.textContent = `${item.name}: ${item.path}`;
      missingList.append(row);
    }
    missingList.hidden = result.missing.length === 0;
    if (!modelReady) {
      runStatus.textContent =
        "Prediction is disabled until the listed private files are supplied.";
    }
  } catch (error) {
    modelReady = false;
    card.classList.add("unavailable");
    title.textContent = "Could not verify the prediction model";
    message.textContent = error.message;
    runButton.disabled = true;
  }
}

function updateSubtype() {
  const er = field("er").value;
  const pr = field("pr").value;
  const her2 = field("her2").value;
  const values = [er, pr, her2];

  if (values.some(value => value === "")) {
    subtypeOutput.textContent = "—";
    manualSubtypeField.hidden = true;
    manualSubtype.required = false;
    return;
  }

  if (!values.every(value => ["Positive", "Negative"].includes(value))) {
    subtypeOutput.textContent = "Select below";
    manualSubtypeField.hidden = false;
    manualSubtype.required = true;
    return;
  }

  manualSubtypeField.hidden = true;
  manualSubtype.required = false;
  manualSubtype.value = "";

  const hr = er === "Positive" || pr === "Positive" ? "HR+" : "HR-";
  const her2Label = her2 === "Positive" ? "HER2+" : "HER2-";
  subtypeOutput.textContent = `${hr}/${her2Label}`;
}

document.querySelectorAll(".receptor").forEach(select => {
  select.addEventListener("change", updateSubtype);
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  imageReady = false;
  preview.replaceChildren();
  preview.hidden = !file;

  if (!file) return;

  if (!["image/png", "image/jpeg"].includes(file.type)) {
    preview.textContent = "Choose a PNG or JPEG image.";
    return;
  }

  const url = URL.createObjectURL(file);
  const image = new Image();
  image.alt = "Selected lymph-node image preview";

  image.onload = () => {
    URL.revokeObjectURL(url);
    preview.replaceChildren(image);

    const dimensions = document.createElement("p");
    dimensions.textContent =
      `${image.naturalWidth} × ${image.naturalHeight} pixels`;

    imageReady =
      image.naturalWidth === 96 && image.naturalHeight === 96;

    const message = document.createElement("p");
    message.textContent = imageReady
      ? "Image dimensions match the model input."
      : "This image needs preparation before the model can use it.";

    preview.append(dimensions, message);
  };

  image.onerror = () => {
    URL.revokeObjectURL(url);
    preview.textContent = "This image could not be opened.";
  };

  image.src = url;
});

function clinicalInput() {
  return {
    age: Number(field("age").value),
    tumour_size: Number(field("tumour-size").value),
    race: field("race").value,
    origin: field("origin").value,
    primary_site: field("primary-site").value,
    histology: field("histology").value.trim(),
    laterality: field("laterality").value,
    grade: field("grade").value,
    er: field("er").value,
    pr: field("pr").value,
    her2: field("her2").value,
    t_stage: field("t-stage").value,
    m_stage: field("m-stage").value,
    breast_subtype: manualSubtype.value,
  };
}

runButton.addEventListener("click", async () => {
  if (!modelReady) {
    runStatus.textContent =
      "Prediction is unavailable because private model files are missing.";
    return;
  }
  const inputs = document.querySelectorAll(
    '#clinical input[required], #clinical select[required]'
  );

  for (const input of inputs) {
    if (!input.reportValidity()) {
      input.focus();
      return;
    }
  }

  if (manualSubtype.required && !manualSubtype.reportValidity()) {
    manualSubtype.focus();
    return;
  }

  if (!imageReady || !imageInput.files[0]) {
    runStatus.textContent = "Choose a readable 96 × 96 image first.";
    return;
  }

  runButton.disabled = true;
  runStatus.textContent = "Running analysis…";

  const form = new FormData();
  form.append("clinical", JSON.stringify(clinicalInput()));
  form.append("image", imageInput.files[0]);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: form,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Analysis failed.");
    }

    const image = result.image_pathway;
    const clinical = result.clinical_pathway;
    latestPrediction = result;

    field("image-result-title").textContent =
      image.predicted_class === "tumour_pattern_detected"
        ? "Tumour pattern detected"
        : "No tumour pattern detected";

    field("image-result-detail").textContent =
      `Image score: ${(image.tumour_pattern_probability * 100).toFixed(4)}%. ` +
      `Threshold: ${(image.threshold * 100).toFixed(1)}%. ` +
      "Result applies to the central region of this patch.";

    field("clinical-result-title").textContent =
      `Predicted ${clinical.predicted_class}`;

    field("clinical-result-detail").textContent =
      `N+ score: ${(clinical.n_positive_probability * 100).toFixed(2)}%. ` +
      `Threshold: ${(clinical.threshold * 100).toFixed(1)}%.`;

    runStatus.textContent = `Analysis complete · model seed ${result.model_seed}`;
    field("result-explanation").hidden = false;
    field("results").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    runStatus.textContent = error.message;
  } finally {
    runButton.disabled = !modelReady;
  }
});

const explainResultsButton = field("explain-results");
const resultExplanationStatus = field("result-explanation-status");

explainResultsButton.addEventListener("click", async () => {
  if (!latestPrediction) return;
  explainResultsButton.disabled = true;
  resultExplanationStatus.textContent =
    "Retrieving evidence and explaining both outputs separately…";

  try {
    const image = latestPrediction.image_pathway;
    const clinical = latestPrediction.clinical_pathway;
    const response = await fetch("/api/explain-results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: {
          predicted_class: image.predicted_class,
          probability: image.tumour_pattern_probability,
        },
        clinical: {
          predicted_class: clinical.predicted_class,
          probability: clinical.n_positive_probability,
        },
        level: field("assistant-level").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Could not explain the results.");
    }

    showAssistantAnswer(result);
    assistantQuestion.value =
      "Explain the latest image and clinical pathway results.";
    resultExplanationStatus.textContent =
      "Explanation is ready in the Knowledge assistant section.";
    field("knowledge").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    resultExplanationStatus.textContent = error.message;
  } finally {
    explainResultsButton.disabled = false;
  }
});

const explainButton = field("explain-background");
const backgroundStatus = field("background-status");
const backgroundContent = field("background-content");
const backgroundText = field("background-text");
const backgroundSources = field("background-sources");

explainButton.addEventListener("click", async () => {
  explainButton.disabled = true;
  backgroundStatus.textContent = "Loading background…";
  backgroundContent.hidden = true;

  try {
    const response = await fetch("/api/background");
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Could not load background.");
    }

    backgroundText.textContent = result.text;
    backgroundSources.replaceChildren();

    for (const source of result.sources) {
      const item = document.createElement("li");
      const link = document.createElement("a");

      link.href = source.url;
      link.textContent = `${source.title} [${source.id}]`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      item.append(link);
      backgroundSources.append(item);
    }

    backgroundContent.hidden = false;
    backgroundStatus.textContent = "Generated background · review sources below";
  } catch (error) {
    backgroundStatus.textContent = error.message;
  } finally {
    explainButton.disabled = false;
  }
});

const askButton = field("ask-assistant");
const assistantQuestion = field("assistant-question");
const assistantStatus = field("assistant-status");
const assistantResponse = field("assistant-response");
const assistantAnswer = field("assistant-answer");
const assistantSources = field("assistant-sources");

async function loadKnowledgeStatus() {
  const row = document.querySelector(".knowledge-status-row");
  const title = field("knowledge-status-title");
  const detail = field("knowledge-status-detail");

  try {
    const response = await fetch("/api/knowledge/status");
    const result = await response.json();
    if (!response.ok || !result.ready) {
      throw new Error("Knowledge database unavailable");
    }
    row.classList.add("ready");
    title.textContent = "Knowledge assistant ready";
    detail.textContent =
      `${result.documents} documents · ${result.chunks} searchable sections`;
  } catch (error) {
    title.textContent = "Knowledge database unavailable";
    detail.textContent = "Run: python -m rag_system.build_database";
  }
}

function showAssistantAnswer(result) {
  assistantAnswer.textContent = result.answer;
  assistantAnswer.dir = "ltr";
  assistantSources.replaceChildren();

  for (const source of result.sources) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = source.url;
    link.textContent = `[${source.citation}] ${source.title} — ${source.heading}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    item.append(link);
    assistantSources.append(item);
  }

  assistantResponse.hidden = false;
  assistantStatus.textContent = result.answer_mode === "general"
    ? "General LLM explanation — verify specialised or current facts."
    : `Evidence-grounded answer from ${result.sources.length} supporting sources.`;
}

askButton.addEventListener("click", async () => {
  const question = assistantQuestion.value.trim();
  if (question.length < 2) {
    assistantStatus.textContent = "Write a question first.";
    assistantQuestion.focus();
    return;
  }

  askButton.disabled = true;
  assistantResponse.hidden = true;
  assistantStatus.textContent = "Searching the verified knowledge base…";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        level: field("assistant-level").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "The assistant could not answer.");
    }

    showAssistantAnswer(result);
  } catch (error) {
    assistantStatus.textContent = error.message;
  } finally {
    askButton.disabled = false;
  }
});

assistantQuestion.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    askButton.click();
  }
});

loadModelStatus();
loadKnowledgeStatus();
