const imageInput = document.querySelector("#image-file");
const preview = document.querySelector("#image-preview");
const runStatus = document.querySelector("#run-status");
const imageRunButton = document.querySelector("#run-image-analysis");

let imageReady = false;
let modelReady = false;
let latestPrediction = null;

function field(id) {
  return document.getElementById(id);
}


async function loadModelStatus() {
  try {
    const response = await fetch("/api/model/status");
    const result = await response.json();
    if (!response.ok) throw new Error("Could not check the model files.");

    modelReady = result.ready;
    imageRunButton.disabled = !modelReady;
  } catch (error) {
    modelReady = false;
    imageRunButton.disabled = true;
  }
}

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

imageRunButton.addEventListener("click", async () => {
  if (!modelReady) {
    runStatus.textContent = "The image model is not ready.";
    return;
  }
  if (!imageReady || !imageInput.files[0]) {
    runStatus.textContent = "Choose a readable 96 × 96 image first.";
    return;
  }
  imageRunButton.disabled = true;
  runStatus.textContent = "Analysing image…";
  field("result-explanation").hidden = true;
  latestPrediction = null;
  try {
    const form = new FormData();
    form.append("image", imageInput.files[0]);
    const response = await fetch("/api/predict-image", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Image analysis failed.");
    latestPrediction = result;
    const image = result.image_pathway;
    field("image-result-title").textContent = image.predicted_class === "tumour_pattern_detected" ? "Tumour pattern detected" : "No tumour pattern detected";
    field("image-result-detail").textContent = `Image score: ${(image.tumour_pattern_probability * 100).toFixed(4)}%. Threshold: ${(image.threshold * 100).toFixed(1)}%. Result applies to the central region of this patch.`;
    runStatus.textContent = "Image analysis complete.";
    field("result-explanation").hidden = false;
    field("results").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    runStatus.textContent = error.message;
  } finally {
    imageRunButton.disabled = !modelReady;
  }
});

const explainResultsButton = field("explain-results");
const resultExplanationStatus = field("result-explanation-status");

explainResultsButton.addEventListener("click", async () => {
  if (!latestPrediction) return;
  explainResultsButton.disabled = true;
  resultExplanationStatus.textContent =
    "Retrieving evidence and explaining the available results…";

  try {
    const image = latestPrediction.image_pathway;
    const response = await fetch("/api/explain-results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: {
          predicted_class: image.predicted_class,
          probability: image.tumour_pattern_probability,
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
      "Explain the latest image result.";
    resultExplanationStatus.textContent =
      "Explanation is ready in the Knowledge assistant section.";
    field("knowledge").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    resultExplanationStatus.textContent = error.message;
  } finally {
    explainResultsButton.disabled = false;
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

  row.classList.remove("ready");
  title.textContent = "Connecting to knowledge assistant";
  detail.textContent = "Checking the local evidence database…";

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch("/api/knowledge/status", {
        cache: "no-store",
      });
      const result = await response.json();
      if (!response.ok || !result.ready) {
        throw new Error("Knowledge database unavailable");
      }
      row.classList.add("ready");
      title.textContent = "Knowledge assistant ready";
      detail.textContent =
        `${result.documents} documents · ${result.chunks} searchable sections`;
      return;
    } catch (error) {
      if (attempt < 3) {
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
    }
  }

  title.textContent = "Cannot reach the knowledge assistant";
  detail.textContent = "Refresh the page after the local server has started.";
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
