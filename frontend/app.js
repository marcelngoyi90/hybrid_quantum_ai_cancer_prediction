const imageInput = document.querySelector("#image-file");
const preview = document.querySelector("#image-preview");
const subtypeOutput = document.querySelector("#subtype");
const manualSubtypeField = document.querySelector("#manual-subtype-field");
const manualSubtype = document.querySelector("#manual-subtype");
const runButton = document.querySelector("#run-analysis");
const runStatus = document.querySelector("#run-status");

let imageReady = false;

function field(id) {
  return document.getElementById(id);
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
    field("results").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    runStatus.textContent = error.message;
  } finally {
    runButton.disabled = false;
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