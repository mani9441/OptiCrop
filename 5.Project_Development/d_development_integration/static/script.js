document.addEventListener("DOMContentLoaded", () => {
  const API_BASE_URL = "http://127.0.0.1:7000";

  // --- VIEW TAB NAVIGATION ROUTING CONTROLLER ---
  const navLinks = document.querySelectorAll(".nav-links li");
  const tabContents = document.querySelectorAll(".tab-content");

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.forEach((item) => item.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      link.classList.add("active");
      const targetId = link.getAttribute("data-target");
      document.getElementById(targetId).classList.add("active");
    });
  });

  // --- PIPELINE DATA UPLOAD INTERCEPT CONTROLS ---
  const fileInput = document.getElementById("pipeline-file");
  const fileMetadata = document.getElementById("file-metadata");
  const uploadBox = document.getElementById("upload-box");

  uploadBox.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileMetadata.textContent = `Mounted: ${fileInput.files[0].name} (${(fileInput.files[0].size / 1024).toFixed(2)} KB)`;
    }
  });

  uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.classList.add("drag-hover");
  });

  uploadBox.addEventListener("dragleave", () => {
    uploadBox.classList.remove("drag-hover");
  });

  uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.classList.remove("drag-hover");
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event("change"));
    }
  });

  // --- INTERFERENCE ENGINE POST 1: RANDOM FOREST CROP RECOMMENDATION ---
  const recommendForm = document.getElementById("recommend-form");
  const recommendResult = document.getElementById("recommend-result");

  recommendForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    recommendResult.classList.add("hide");

    const formData = new FormData(recommendForm);
    const payload = {
      N: parseInt(formData.get("N")),
      P: parseInt(formData.get("P")),
      K: parseInt(formData.get("K")),
      temperature: parseFloat(formData.get("temperature")),
      humidity: parseFloat(formData.get("humidity")),
      ph: parseFloat(formData.get("ph")),
      rainfall: parseFloat(formData.get("rainfall")),
    };

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const resData = await response.json();

      if (resData.success) {
        recommendResult.className = "result-box success";
        recommendResult.innerHTML = `<i class="fa-solid fa-square-poll-horizontal"></i> Optimal Target Classification: <strong>${resData.predicted_crop.toUpperCase()}</strong>`;
      } else {
        throw new Error(
          resData.error || "Inference exception state encountered",
        );
      }
    } catch (err) {
      recommendResult.className = "result-box danger";
      recommendResult.textContent = `Error: ${err.message}`;
    }
    recommendResult.classList.remove("hide");
  });

  // --- INTERFERENCE ENGINE POST 2: OPERATIONALharvest PLAN VALIDATION ---
  const validateForm = document.getElementById("validate-form");
  const validateResult = document.getElementById("validate-result");

  validateForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    validateResult.classList.add("hide");

    const formData = new FormData(validateForm);
    const payload = {
      crop: formData.get("crop"),
      N: parseInt(formData.get("N")),
      P: parseInt(formData.get("P")),
      K: parseInt(formData.get("K")),
      temperature: parseFloat(formData.get("temperature")),
      humidity: parseFloat(formData.get("humidity")),
      ph: parseFloat(formData.get("ph")),
      rainfall: parseFloat(formData.get("rainfall")),
    };

    try {
      const response = await fetch(`${API_BASE_URL}/validate-crop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const resData = await response.json();

      if (resData.success) {
        validateResult.className = resData.is_best_choice
          ? "result-box success"
          : "result-box danger";
        validateResult.innerHTML = `
                    <i class="${resData.is_best_choice ? "fa-solid fa-circle-check" : "fa-solid fa-triangle-exclamation"}"></i> 
                    <strong>${resData.message}</strong><br>
                    <small>Engine Recommendation: ${resData.recommended_crop} | Your Intent Selection: ${resData.planned_crop}</small>
                `;
      } else {
        throw new Error(
          resData.error || "Validation exception state encountered",
        );
      }
    } catch (err) {
      validateResult.className = "result-box danger";
      validateResult.textContent = `Error: ${err.message}`;
    }
    validateResult.classList.remove("hide");
  });

  // --- INTERFERENCE ENGINE POST 3: COMPLEX PIPELINE RESEARCH DATA MATRIX ---
  const analyticsForm = document.getElementById("analytics-form");
  const pipelineLoading = document.getElementById("pipeline-loading");
  const pipelineResults = document.getElementById("pipeline-results");
  const pcaTableBody = document.querySelector("#pca-table tbody");
  const chartsDisplayGrid = document.getElementById("charts-display-grid");
  const qualityMetrics = document.getElementById("quality-metrics");
  const sampleDatasetBtn = document.getElementById("sample-dataset-btn");
  let analyticsRunning = false;

  async function executeResearchPipeline(submissionData) {
    if (analyticsRunning) return;

    analyticsRunning = true;

    const runBtn = analyticsForm.querySelector('button[type="submit"]');
    const originalRunText = runBtn.innerHTML;
    const originalSampleText = sampleDatasetBtn.innerHTML;

    runBtn.disabled = true;
    sampleDatasetBtn.disabled = true;

    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';
    sampleDatasetBtn.innerHTML =
      '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

    pipelineLoading.classList.remove("hide");
    pipelineResults.classList.add("hide");

    try {
      const response = await fetch(`${API_BASE_URL}/research-analytics`, {
        method: "POST",
        body: submissionData,
      });

      const payload = await response.json();

      if (!response.ok || !payload.success) {
        throw new Error(payload.error || "Analytics execution failed.");
      }

      const dataContext = payload.analytics || payload;

      if (dataContext.quality) {
        qualityMetrics.innerHTML = `
        <table class="data-table">
          <tr>
            <td>Categorical Features</td>
            <td><strong>${dataContext.quality["Categorical Features"] ?? "-"}</strong></td>
          </tr>
          <tr>
            <td>Numeric Features</td>
            <td><strong>${dataContext.quality["Numeric Features"] ?? "-"}</strong></td>
          </tr>
          <tr>
            <td>Missing Values</td>
            <td><strong>${dataContext.quality["Missing Values"] ?? "-"}</strong></td>
          </tr>
          <tr>
            <td>Total Rows</td>
            <td><strong>${dataContext.quality["Total Rows"] ?? "-"}</strong></td>
          </tr>
        </table>
      `;
      }

      pcaTableBody.innerHTML = "";

      if (dataContext.pca && dataContext.pca.coordinates) {
        dataContext.pca.coordinates.slice(0, 5).forEach((row) => {
          const tr = document.createElement("tr");

          tr.innerHTML = `
          <td>
            <span class="alert-success-banner" style="padding:2px 6px;">
              ${row.label}
            </span>
          </td>
          <td>${row.PC1.toFixed(4)}</td>
          <td>${row.PC2.toFixed(4)}</td>
        `;

          pcaTableBody.appendChild(tr);
        });
      }

      chartsDisplayGrid.innerHTML = "";

      if (payload.charts && Object.keys(payload.charts).length > 0) {
        const renderCharts = (obj, prefix = "") => {
          Object.entries(obj).forEach(([key, value]) => {
            const title = prefix ? `${prefix} - ${key}` : key;

            if (typeof value === "string" && value.length > 100) {
              appendChartCard(title, value);
            } else if (
              value &&
              typeof value === "object" &&
              !Array.isArray(value)
            ) {
              renderCharts(value, title);
            }
          });
        };

        renderCharts(payload.charts);
      } else {
        chartsDisplayGrid.innerHTML = `
        <div class="empty-state">
          <i class="fa-solid fa-chart-column"></i>
          <p>No charts were returned by the analytics engine.</p>
        </div>
      `;
      }

      pipelineResults.classList.remove("hide");
    } catch (err) {
      alert(`Pipeline Exception State: ${err.message}`);
    } finally {
      analyticsRunning = false;

      runBtn.disabled = false;
      sampleDatasetBtn.disabled = false;

      runBtn.innerHTML = originalRunText;
      sampleDatasetBtn.innerHTML = originalSampleText;

      pipelineLoading.classList.add("hide");
    }
  }

  analyticsForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (fileInput.files.length === 0) {
      return alert(
        "Please select a CSV/XLSX dataset or use the Sample Dataset button.",
      );
    }

    const submissionData = new FormData();
    submissionData.append("file", fileInput.files[0]);

    const customName = document.getElementById("dataset_name").value.trim();

    if (customName) {
      submissionData.append("dataset_name", customName);
    }

    await executeResearchPipeline(submissionData);
  });

  sampleDatasetBtn.addEventListener("click", async () => {
    const submissionData = new FormData();

    submissionData.append("use_sample", "true");

    const customName = document.getElementById("dataset_name").value.trim();

    if (customName) {
      submissionData.append("dataset_name", customName);
    }

    await executeResearchPipeline(submissionData);
  });

  function appendChartCard(title, imageData) {
    const wrapper = document.createElement("div");
    wrapper.className = "chart-wrapper-card";

    const src = imageData.startsWith("data:")
      ? imageData
      : `data:image/png;base64,${imageData}`;

    wrapper.innerHTML = `
    <h4>${title.replace(/_/g, " ").replace(/-/g, " ")}</h4>
    <img
      src="${src}"
      alt="${title}"
      loading="lazy"
    >
  `;

    chartsDisplayGrid.appendChild(wrapper);
  }
});
