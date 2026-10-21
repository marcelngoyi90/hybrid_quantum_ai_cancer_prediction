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

    if (!response.ok) {
      throw new Error("Could not check the model files.");
    }

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

  if (!file) {
    return;
  }

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
      image.naturalWidth === 96
      && image.naturalHeight === 96;

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

    const response = await fetch("/api/predict-image", {
      method: "POST",
      body: form,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.detail || "Image analysis failed.",
      );
    }

    latestPrediction = result;

    const image = result.image_pathway;

    field("image-result-title").textContent =
      image.predicted_class === "tumour_pattern_detected"
        ? "Tumour pattern detected"
        : "No tumour pattern detected";

    field("image-result-detail").textContent =
      `Image score: ${
        (image.tumour_pattern_probability * 100).toFixed(4)
      }%. Threshold: ${
        (image.threshold * 100).toFixed(1)
      }%. Result applies to the central region of this patch.`;

    runStatus.textContent = "Image analysis complete.";

    field("result-explanation").hidden = false;

    field("results").scrollIntoView({
      behavior: "smooth",
    });
  } catch (error) {
    runStatus.textContent = error.message;
  } finally {
    imageRunButton.disabled = !modelReady;
  }
});

const explainResultsButton = field("explain-results");
const resultExplanationStatus = field(
  "result-explanation-status",
);

const askButton = field("ask-assistant");
const assistantQuestion = field("assistant-question");
const assistantStatus = field("assistant-status");
const assistantResponse = field("assistant-response");
const assistantAnswer = field("assistant-answer");
const assistantAnswerHeading = field(
  "assistant-answer-heading",
);
const assistantSources = field("assistant-sources");

const listenAnswerButton = field("listen-answer");
const stopAnswerButton = field("stop-answer");
const audioStatus = field("audio-status");

const speechSupported =
  "speechSynthesis" in window
  && "SpeechSynthesisUtterance" in window;

let activeUtterance = null;
let speechSession = 0;
let speechChunks = [];
let speechChunkIndex = 0;
let speechState = "idle";
let speechStartTimer = null;

function splitAnswerForSpeech(text) {
  const sentences = text.match(
    /[^.!?]+[.!?]+|[^.!?]+$/g,
  ) || [text];

  const chunks = [];
  let currentChunk = "";

  for (const sentence of sentences) {
    const cleanSentence = sentence.trim();

    if (!cleanSentence) {
      continue;
    }

    const candidate = currentChunk
      ? `${currentChunk} ${cleanSentence}`
      : cleanSentence;

    if (currentChunk && candidate.length > 260) {
      chunks.push(currentChunk);
      currentChunk = cleanSentence;
    } else {
      currentChunk = candidate;
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks.length ? chunks : [text];
}

function setAudioReadyState(message = "") {
  listenAnswerButton.textContent = "Listen to answer";

  listenAnswerButton.setAttribute(
    "aria-pressed",
    "false",
  );

  listenAnswerButton.disabled =
    !assistantAnswer.textContent.trim();

  stopAnswerButton.disabled = true;
  audioStatus.textContent = message;
}

function resetSpeechControls(message = "") {
  speechSession += 1;
  speechState = "idle";
  speechChunks = [];
  speechChunkIndex = 0;
  activeUtterance = null;

  window.clearTimeout(speechStartTimer);

  if (!speechSupported) {
    listenAnswerButton.disabled = true;
    stopAnswerButton.disabled = true;

    audioStatus.textContent =
      "Audio playback is not supported in this browser.";

    return;
  }

  window.speechSynthesis.cancel();

  setAudioReadyState(message);
}

function speakNextChunk(session) {
  if (
    session !== speechSession
    || speechState !== "playing"
  ) {
    return;
  }

  if (speechChunkIndex >= speechChunks.length) {
    activeUtterance = null;
    speechState = "idle";

    setAudioReadyState(
      "Finished reading the answer.",
    );

    return;
  }

  const utterance = new SpeechSynthesisUtterance(
    speechChunks[speechChunkIndex],
  );

  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;

  utterance.onend = () => {
    if (
      session !== speechSession
      || speechState !== "playing"
    ) {
      return;
    }

    speechChunkIndex += 1;

    window.setTimeout(() => {
      speakNextChunk(session);
    }, 80);
  };

  utterance.onerror = () => {
    if (session !== speechSession) {
      return;
    }

    activeUtterance = null;
    speechState = "idle";

    setAudioReadyState(
      "Audio playback could not be started.",
    );
  };

  activeUtterance = utterance;

  window.speechSynthesis.speak(utterance);
}

function speakOrPauseAnswer() {
  const answerText = assistantAnswer.textContent.trim();

  if (!speechSupported || !answerText) {
    return;
  }

  if (speechState === "paused") {
    speechState = "playing";

    window.speechSynthesis.resume();

    listenAnswerButton.textContent = "Pause audio";

    listenAnswerButton.setAttribute(
      "aria-pressed",
      "true",
    );

    audioStatus.textContent =
      "Reading the answer aloud.";

    return;
  }

  if (speechState === "playing") {
    speechState = "paused";

    window.speechSynthesis.pause();

    listenAnswerButton.textContent = "Resume audio";

    listenAnswerButton.setAttribute(
      "aria-pressed",
      "false",
    );

    audioStatus.textContent = "Audio paused.";

    return;
  }

  speechSession += 1;

  const session = speechSession;

  window.speechSynthesis.cancel();

  speechChunks = splitAnswerForSpeech(answerText);
  speechChunkIndex = 0;
  speechState = "starting";

  listenAnswerButton.disabled = true;
  stopAnswerButton.disabled = true;

  audioStatus.textContent = "Preparing audio…";

  speechStartTimer = window.setTimeout(() => {
    if (
      session !== speechSession
      || speechState !== "starting"
    ) {
      return;
    }

    speechState = "playing";

    listenAnswerButton.disabled = false;
    listenAnswerButton.textContent = "Pause audio";

    listenAnswerButton.setAttribute(
      "aria-pressed",
      "true",
    );

    stopAnswerButton.disabled = false;

    audioStatus.textContent =
      "Reading the answer aloud.";

    speakNextChunk(session);
  }, 350);
}

function stopAnswerAudio() {
  resetSpeechControls("Audio stopped.");
}

async function loadKnowledgeStatus() {
  const row = document.querySelector(
    ".knowledge-status-row",
  );

  const title = field("knowledge-status-title");
  const detail = field("knowledge-status-detail");

  row.classList.remove("ready");

  title.textContent =
    "Connecting to knowledge assistant";

  detail.textContent =
    "Checking the local evidence database…";

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(
        "/api/knowledge/status",
        {
          cache: "no-store",
        },
      );

      const result = await response.json();

      if (!response.ok || !result.ready) {
        throw new Error("Knowledge database unavailable");
      }

      row.classList.add("ready");

      title.textContent = "Knowledge assistant ready";

      detail.textContent =
        `${result.documents} documents · `
        + `${result.chunks} searchable sections`;

      return;
    } catch (error) {
      if (attempt < 3) {
        await new Promise((resolve) => {
          window.setTimeout(resolve, 800);
        });
      }
    }
  }

  title.textContent =
    "Cannot reach the knowledge assistant";

  detail.textContent =
    "Refresh the page after the local server has started.";
}

function showAssistantAnswer(
  result,
  imageResult = null,
) {
  resetSpeechControls("");

  const resultContext = field(
    "assistant-result-context",
  );

  if (imageResult) {
    const tumourDetected =
      imageResult.predicted_class
      === "tumour_pattern_detected";

    field("assistant-result-title").textContent =
      tumourDetected
        ? "Tumour pattern detected"
        : "No tumour pattern detected";

    resultContext.hidden = false;

    assistantAnswerHeading.textContent = "Explanation";
  } else {
    resultContext.hidden = true;

    assistantAnswerHeading.textContent = "Answer";
  }

  assistantAnswer.textContent = result.answer;
  assistantAnswer.dir = "ltr";

  if (speechSupported) {
    listenAnswerButton.disabled = false;
    audioStatus.textContent = "Audio is ready.";
  }

  assistantSources.replaceChildren();

  for (const source of result.sources) {
    const item = document.createElement("li");
    const link = document.createElement("a");

    link.href = source.url;

    link.textContent =
      `[${source.citation}] ${source.title}`
      + ` — ${source.heading}`;

    link.target = "_blank";
    link.rel = "noopener noreferrer";

    item.append(link);
    assistantSources.append(item);
  }

  assistantResponse.hidden = false;

  assistantStatus.textContent =
    result.answer_mode === "general"
      ? "General LLM explanation — verify specialised or current facts."
      : (
        `Evidence-grounded answer from `
        + `${result.sources.length} supporting sources.`
      );
}

explainResultsButton.addEventListener("click", async () => {
  if (!latestPrediction) {
    return;
  }

  resetSpeechControls("");

  explainResultsButton.disabled = true;

  resultExplanationStatus.textContent =
    "Retrieving evidence and explaining the available results…";

  try {
    const image = latestPrediction.image_pathway;

    const response = await fetch("/api/explain-results", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
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
      throw new Error(
        result.detail || "Could not explain the results.",
      );
    }

    showAssistantAnswer(result, image);

    assistantQuestion.value =
      "Explain the latest image result.";

    resultExplanationStatus.textContent =
      "Explanation is ready in the Knowledge assistant section.";

    field("knowledge").scrollIntoView({
      behavior: "smooth",
    });
  } catch (error) {
    resultExplanationStatus.textContent = error.message;
  } finally {
    explainResultsButton.disabled = false;
  }
});

askButton.addEventListener("click", async () => {
  const question = assistantQuestion.value.trim();

  if (question.length < 2) {
    assistantStatus.textContent =
      "Write a question first.";

    assistantQuestion.focus();

    return;
  }

  resetSpeechControls("");

  askButton.disabled = true;
  assistantResponse.hidden = true;

  assistantStatus.textContent =
    "Searching the verified knowledge base…";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        level: field("assistant-level").value,
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.detail || "The assistant could not answer.",
      );
    }

    showAssistantAnswer(result);
  } catch (error) {
    assistantStatus.textContent = error.message;
  } finally {
    askButton.disabled = false;
  }
});

listenAnswerButton.addEventListener(
  "click",
  speakOrPauseAnswer,
);

stopAnswerButton.addEventListener(
  "click",
  stopAnswerAudio,
);

assistantQuestion.addEventListener("keydown", (event) => {
  if (
    (event.ctrlKey || event.metaKey)
    && event.key === "Enter"
  ) {
    event.preventDefault();
    askButton.click();
  }
});

loadModelStatus();
loadKnowledgeStatus();