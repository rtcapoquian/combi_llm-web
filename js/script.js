// Eco-Sort AI JavaScript Application
class EcoSortAI {
  constructor() {
    this.camera = null;
    this.canvas = null;
    this.context = null;
    this.isCapturing = false;
    this.recyclingList = JSON.parse(
      localStorage.getItem("recyclingList") || "[]"
    );

    // API endpoints (adjust these to match your backend)
    this.endpoints = {
      yoloInference: "/api/classify",
      llmGuidance: "/api/guidance",
      addToList: "/api/recycling/add",
    };

    this.init();
  }

  init() {
    this.setupElements();
    this.setupEventListeners();
    this.updateRecyclingDisplay();
    this.updateStatus("yoloStatus", "YOLO Ready", "success");
    this.updateStatus("llmStatus", "LLM Ready", "success");
  }

  setupElements() {
    // Camera elements
    this.camera = document.getElementById("camera");
    this.canvas = document.getElementById("canvas");
    this.context = this.canvas.getContext("2d");

    // Control elements
    this.startCameraBtn = document.getElementById("startCamera");
    this.captureBtn = document.getElementById("captureBtn");
    this.uploadBtn = document.getElementById("uploadBtn");
    this.fileInput = document.getElementById("fileInput");

    // Result containers
    this.detectionResults = document.getElementById("detectionResults");
    this.guidanceResults = document.getElementById("guidanceResults");
    this.recyclingListContainer = document.getElementById("recyclingList");

    // Status elements
    this.yoloStatus = document.getElementById("yoloStatus");
    this.llmStatus = document.getElementById("llmStatus");
    this.processingTime = document.getElementById("processingTime");

    // Modal elements
    this.loadingOverlay = document.getElementById("loadingOverlay");
    this.loadingStatus = document.getElementById("loadingStatus");
    this.progressFill = document.getElementById("progressFill");
    this.errorModal = document.getElementById("errorModal");
    this.errorMessage = document.getElementById("errorMessage");
  }

  setupEventListeners() {
    // Camera controls
    this.startCameraBtn.addEventListener("click", () => this.startCamera());
    this.captureBtn.addEventListener("click", () => this.captureAndAnalyze());
    this.uploadBtn.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => this.handleFileUpload(e));

    // Recycling list controls
    document
      .getElementById("clearList")
      .addEventListener("click", () => this.clearRecyclingList());
    document
      .getElementById("exportList")
      .addEventListener("click", () => this.exportRecyclingList());

    // Modal controls
    document
      .getElementById("closeError")
      .addEventListener("click", () => this.hideError());
    document
      .getElementById("okError")
      .addEventListener("click", () => this.hideError());

    // Close modal when clicking outside
    this.errorModal.addEventListener("click", (e) => {
      if (e.target === this.errorModal) this.hideError();
    });
  }

  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      this.camera.srcObject = stream;
      this.startCameraBtn.textContent = "Camera Active";
      this.startCameraBtn.disabled = true;
      this.captureBtn.disabled = false;

      this.updateStatus("yoloStatus", "Camera Active", "success");
    } catch (error) {
      console.error("Error accessing camera:", error);
      this.showError(
        "Unable to access camera. Please check permissions and try again."
      );
    }
  }

  async captureAndAnalyze() {
    if (this.isCapturing) return;

    this.isCapturing = true;
    this.showLoading("Capturing image...");

    try {
      // Capture image from camera
      this.canvas.width = this.camera.videoWidth;
      this.canvas.height = this.camera.videoHeight;
      this.context.drawImage(this.camera, 0, 0);

      const imageBlob = await this.canvasToBlob();
      await this.analyzeImage(imageBlob);
    } catch (error) {
      console.error("Error capturing image:", error);
      this.showError("Failed to capture image. Please try again.");
    } finally {
      this.isCapturing = false;
      this.hideLoading();
    }
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      this.showError("Please select a valid image file.");
      return;
    }

    try {
      await this.analyzeImage(file);
    } catch (error) {
      console.error("Error analyzing uploaded image:", error);
      this.showError("Failed to analyze uploaded image. Please try again.");
    }

    // Clear file input
    event.target.value = "";
  }

  canvasToBlob() {
    return new Promise((resolve) => {
      this.canvas.toBlob(resolve, "image/jpeg", 0.8);
    });
  }

  async analyzeImage(imageBlob) {
    const startTime = Date.now();

    try {
      this.showLoading("Running YOLO inference...");
      this.updateProgress(20);

      // Step 1: YOLO Classification
      const detections = await this.runYOLOInference(imageBlob);
      this.updateProgress(50);

      if (detections && detections.length > 0) {
        this.displayDetectionResults(detections);

        this.updateLoadingStatus("Getting AI guidance...");
        this.updateProgress(70);

        // Step 2: LLM Guidance
        const guidance = await this.getLLMGuidance(detections);
        this.updateProgress(90);

        this.displayGuidanceResults(guidance);

        // Step 3: Add to recycling list
        await this.addDetectionsToList(detections);
        this.updateProgress(100);
      } else {
        this.displayNoDetections();
      }

      const processingTime = Date.now() - startTime;
      this.updateStatus("processingTime", `${processingTime}ms`, "info");
    } catch (error) {
      console.error("Error analyzing image:", error);
      this.showError(`Analysis failed: ${error.message}`);
    } finally {
      this.hideLoading();
    }
  }

  async runYOLOInference(imageBlob) {
    try {
      // Create form data for the image
      const formData = new FormData();
      formData.append("image", imageBlob);

      // Call your Python backend
      const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Classification failed");
      }

      // Update processing time display (now in seconds)
      if (typeof data.processing_time_s === "number") {
        this.updateStatus(
          "processingTime",
          `${data.processing_time_s.toFixed(2)}s`,
          "info"
        );
      }

      return data.detections || [];
    } catch (error) {
      console.error("YOLO inference error:", error);
      // Fall back to simulation if backend is not available
      console.log("Falling back to simulation mode...");
      const response = await this.simulateYOLOInference(imageBlob);
      return response.detections || [];
    }
  }

  async simulateYOLOInference(imageBlob) {
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Simulate detection results based on common waste items
    const mockDetections = [
      {
        class: "plastic_bottle",
        confidence: 0.89,
        bbox: [120, 80, 200, 180],
        category: "Recyclable Plastic",
      },
      {
        class: "electronic_device",
        confidence: 0.76,
        bbox: [250, 100, 350, 200],
        category: "E-Waste",
      },
    ];

    // Randomly return 0-2 detections for demo
    const numDetections = Math.floor(Math.random() * 3);
    return {
      detections: mockDetections.slice(0, numDetections),
    };
  }

  async getLLMGuidance(detections) {
    try {
      // Call your Python backend for LLM guidance
      const response = await fetch("/api/guidance", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ detections: detections }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Guidance request failed");
      }

      return data.guidance || [];
    } catch (error) {
      console.error("LLM guidance error:", error);
      // Fall back to simulation if backend is not available
      console.log("Falling back to simulated guidance...");
      const guidance = [];

      for (const detection of detections) {
        const itemGuidance = await this.simulateLLMGuidance(detection);
        guidance.push(itemGuidance);
      }

      return guidance;
    }
  }

  async simulateLLMGuidance(detection) {
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 500));

    const guidanceData = {
      plastic_bottle: {
        title: "Plastic Bottle Recycling Guide",
        content: `This plastic bottle can be recycled! Here's what you need to know:

🔍 **Classification**: This appears to be a PET plastic bottle, which is highly recyclable.

♻️ **Preparation Steps**:
• Remove the cap and label if possible
• Rinse with water to remove any residue
• Compress to save space in recycling bin

📍 **Disposal**: Place in your regular recycling bin or take to any bottle deposit location.

🌱 **Environmental Impact**: Recycling this bottle saves energy and reduces plastic waste in landfills.`,
        disassembly: [
          "Remove cap (often made of different plastic)",
          "Peel off label if easily removable",
          "Rinse interior with water",
          "Compress bottle to save space",
        ],
        safety: [
          "No special safety precautions needed",
          "Ensure bottle is empty before processing",
        ],
      },
      electronic_device: {
        title: "Electronic Device Disposal Guide",
        content: `This electronic device requires special handling for proper recycling:

⚠️ **Important**: Electronic devices contain valuable materials but also potentially hazardous substances.

🔧 **Preparation**:
• Remove batteries if possible
• Wipe any personal data if applicable
• Keep original cables and accessories together

📍 **Disposal**: Take to certified e-waste recycling facility. Do not put in regular trash.

💡 **Value Recovery**: Contains precious metals like gold, silver, and rare earth elements that can be recovered.`,
        disassembly: [
          "Power off completely",
          "Remove batteries if accessible",
          "Separate cables and accessories",
          "Keep original packaging if available",
        ],
        safety: [
          "Do not attempt to open sealed devices",
          "Handle with care to avoid cuts from broken parts",
          "Keep away from children",
        ],
      },
    };

    const guidance = guidanceData[detection.class] || {
      title: `${detection.class} Disposal Guide`,
      content: `This item has been classified as ${detection.category}. Please consult local waste management guidelines for proper disposal procedures.`,
      disassembly: ["Consult local recycling guidelines"],
      safety: ["Follow standard waste handling precautions"],
    };

    return {
      detection: detection,
      ...guidance,
    };
  }

  displayDetectionResults(detections) {
    const container = this.detectionResults;
    container.innerHTML = "";

    detections.forEach((detection, index) => {
      const detectionDiv = document.createElement("div");
      detectionDiv.className = "detection-item fade-in";
      detectionDiv.innerHTML = `
                <div class="detection-header">
                    <span class="detection-label">${detection.class.replace(
                      "_",
                      " "
                    )}</span>
                    <span class="detection-confidence">${(
                      detection.confidence * 100
                    ).toFixed(1)}%</span>
                </div>
                <div class="detection-coords">
                    Location: [${detection.bbox.join(", ")}]
                </div>
                <div style="margin-top: 8px;">
                    <span class="item-category">${detection.category}</span>
                </div>
            `;

      container.appendChild(detectionDiv);

      // Add detection box overlay (if camera is active)
      this.showDetectionBox(detection.bbox, index);
    });

    // Display annotated image if available
    this.displayAnnotatedImage(detections);

    this.updateStatus(
      "yoloStatus",
      `${detections.length} items detected`,
      "success"
    );
  }

  displayAnnotatedImage(detections) {
    if (detections && detections.length > 0) {
      // Check if we have an annotated image path from the backend
      const firstDetection = detections[0];
      if (firstDetection.annotated_image_path) {
        const annotatedSection = document.getElementById(
          "annotatedImageSection"
        );
        const annotatedImage = document.getElementById("annotatedImage");

        if (annotatedSection && annotatedImage) {
          // Construct the URL for the annotated image
          const imageUrl = `/annotated_images/${firstDetection.annotated_image_path}`;

          // Set the image source and show the section
          annotatedImage.src = imageUrl;
          annotatedImage.onload = () => {
            annotatedSection.style.display = "block";
            console.log("Annotated image loaded successfully");
          };
          annotatedImage.onerror = () => {
            console.error("Failed to load annotated image");
            annotatedSection.style.display = "none";
          };
        }
      }
    }
  }

  displayNoDetections() {
    this.detectionResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <p>No waste items detected in this image</p>
                <small>Try capturing a clearer image or adjusting lighting</small>
            </div>
        `;
    this.guidanceResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-robot"></i>
                <p>No guidance available - no items detected</p>
            </div>
        `;
  }

  displayGuidanceResults(guidanceList) {
    const container = this.guidanceResults;
    container.innerHTML = "";

    guidanceList.forEach((guidance) => {
      const guidanceDiv = document.createElement("div");
      guidanceDiv.className = "guidance-content fade-in";

      // Build HTML content conditionally
      let htmlContent = "";

      // Only show title if it exists and is not empty
      if (guidance.title && guidance.title.trim() !== "") {
        htmlContent += `<h4><i class="fas fa-lightbulb"></i> ${guidance.title}</h4>`;
      }

      // Always show content
      htmlContent += `<div class="guidance-text">${guidance.content.replace(
        /\n/g,
        "<br>"
      )}</div>`;

      // Only show disassembly section if it exists and has items
      if (guidance.disassembly && guidance.disassembly.length > 0) {
        htmlContent += `
                <h5><i class="fas fa-tools"></i> Disassembly Steps:</h5>
                <ul class="guidance-list">
                    ${guidance.disassembly
                      .map((step) => `<li>${step}</li>`)
                      .join("")}
                </ul>`;
      }

      // Only show safety section if it exists and has items
      if (guidance.safety && guidance.safety.length > 0) {
        htmlContent += `
                <h5><i class="fas fa-shield-alt"></i> Safety Precautions:</h5>
                <ul class="guidance-list">
                    ${guidance.safety
                      .map((safety) => `<li>${safety}</li>`)
                      .join("")}
                </ul>`;
      }

      guidanceDiv.innerHTML = htmlContent;
      container.appendChild(guidanceDiv);
    });

    this.updateStatus(
      "llmStatus",
      `Guidance for ${guidanceList.length} items`,
      "success"
    );
  }

  showDetectionBox(bbox, index) {
    const [x1, y1, x2, y2] = bbox;
    const detectionBox = document.getElementById("detectionBox");

    // Calculate relative position (assuming normalized coordinates)
    const cameraRect = this.camera.getBoundingClientRect();
    const boxWidth = Math.abs(x2 - x1);
    const boxHeight = Math.abs(y2 - y1);

    detectionBox.style.left = `${x1}px`;
    detectionBox.style.top = `${y1}px`;
    detectionBox.style.width = `${boxWidth}px`;
    detectionBox.style.height = `${boxHeight}px`;
    detectionBox.classList.add("active");

    // Hide after 3 seconds
    setTimeout(() => {
      detectionBox.classList.remove("active");
    }, 3000);
  }

  async addDetectionsToList(detections) {
    for (const detection of detections) {
      try {
        // Add to backend recycling list
        const response = await fetch("/api/recycling/add", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            item_name: detection.class.replace("_", " "),
            category: detection.category,
            quantity: 1,
            notes: `Detected with ${(detection.confidence * 100).toFixed(
              1
            )}% confidence`,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            // Update local list with backend response
            this.recyclingList = data.recycling_list || [];
          }
        }
      } catch (error) {
        console.warn("Failed to add to backend list, adding locally:", error);
        // Fall back to local storage
        const item = {
          id: Date.now() + Math.random(),
          name: detection.class.replace("_", " "),
          category: detection.category,
          quantity: 1,
          confidence: detection.confidence,
          addedAt: new Date().toISOString(),
        };

        this.recyclingList.push(item);
      }
    }

    this.saveRecyclingList();
    this.updateRecyclingDisplay();
  }

  updateRecyclingDisplay() {
    const container = this.recyclingListContainer;

    if (this.recyclingList.length === 0) {
      container.innerHTML = `
                <div class="no-items">
                    <i class="fas fa-recycle"></i>
                    <p>No items in recycling list yet</p>
                </div>
            `;
      return;
    }

    container.innerHTML = this.recyclingList
      .map(
        (item) => `
            <div class="recycling-item fade-in">
                <div class="item-info">
                    <div class="item-name">${
                      item.item_name || item.name || "Unknown Item"
                    }</div>
                    <div class="item-category">${
                      item.category || "General Waste"
                    }</div>
                    ${
                      item.confidence
                        ? `<small>Confidence: ${(item.confidence * 100).toFixed(
                            1
                          )}%</small>`
                        : ""
                    }
                    ${
                      item.notes
                        ? `<small class="item-notes">${item.notes}</small>`
                        : ""
                    }
                </div>
                <div class="item-quantity">${item.quantity || 1}</div>
            </div>
        `
      )
      .join("");
  }

  clearRecyclingList() {
    this.recyclingList = [];
    this.saveRecyclingList();
    this.updateRecyclingDisplay();
  }

  exportRecyclingList() {
    if (this.recyclingList.length === 0) {
      this.showError("No items to export");
      return;
    }

    const data = {
      exportDate: new Date().toISOString(),
      totalItems: this.recyclingList.length,
      items: this.recyclingList,
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `recycling-list-${
      new Date().toISOString().split("T")[0]
    }.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
  }

  saveRecyclingList() {
    localStorage.setItem("recyclingList", JSON.stringify(this.recyclingList));
  }

  updateStatus(elementId, text, type = "info") {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = text;
      element.className = `status-${type}`;
    }
  }

  showLoading(message = "Processing...") {
    this.loadingOverlay.classList.add("active");
    this.updateLoadingStatus(message);
    this.updateProgress(0);
  }

  hideLoading() {
    this.loadingOverlay.classList.remove("active");
  }

  updateLoadingStatus(message) {
    this.loadingStatus.textContent = message;
  }

  updateProgress(percent) {
    this.progressFill.style.width = `${percent}%`;
  }

  showError(message) {
    this.errorMessage.textContent = message;
    this.errorModal.classList.add("active");
  }

  hideError() {
    this.errorModal.classList.remove("active");
  }
}

// Additional utility functions
class APIClient {
  constructor(baseURL = "") {
    this.baseURL = baseURL;
  }

  async post(endpoint, data) {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      method: "POST",
      body: data,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async get(endpoint) {
    const response = await fetch(`${this.baseURL}${endpoint}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

// Performance monitoring
class PerformanceMonitor {
  constructor() {
    this.metrics = {};
  }

  startTiming(key) {
    this.metrics[key] = { start: performance.now() };
  }

  endTiming(key) {
    if (this.metrics[key]) {
      this.metrics[key].end = performance.now();
      this.metrics[key].duration =
        this.metrics[key].end - this.metrics[key].start;
    }
  }

  getTiming(key) {
    return this.metrics[key]?.duration || 0;
  }

  getReport() {
    const report = {};
    Object.keys(this.metrics).forEach((key) => {
      report[key] = this.getTiming(key);
    });
    return report;
  }
}

// Integration helper for connecting to Python backend
class PythonBackendIntegration {
  constructor(ecoSortInstance) {
    this.ecoSort = ecoSortInstance;
    this.wsConnection = null;
    this.setupWebSocket();
  }

  setupWebSocket() {
    // WebSocket connection for real-time communication
    try {
      this.wsConnection = new WebSocket("ws://localhost:8765");

      this.wsConnection.onopen = () => {
        console.log("Connected to Python backend");
        this.ecoSort.updateStatus(
          "llmStatus",
          "Connected to backend",
          "success"
        );
      };

      this.wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleBackendMessage(data);
      };

      this.wsConnection.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.ecoSort.updateStatus(
          "llmStatus",
          "Backend connection failed",
          "error"
        );
      };

      this.wsConnection.onclose = () => {
        console.log("Disconnected from Python backend");
        this.ecoSort.updateStatus(
          "llmStatus",
          "Backend disconnected",
          "warning"
        );
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.setupWebSocket(), 5000);
      };
    } catch (error) {
      console.error("Failed to setup WebSocket:", error);
    }
  }

  handleBackendMessage(data) {
    switch (data.type) {
      case "detection_result":
        this.ecoSort.displayDetectionResults(data.detections);
        break;
      case "guidance_result":
        this.ecoSort.displayGuidanceResults(data.guidance);
        break;
      case "status_update":
        this.ecoSort.updateLoadingStatus(data.message);
        this.ecoSort.updateProgress(data.progress);
        break;
      default:
        console.log("Unknown message type:", data.type);
    }
  }

  async sendImage(imageBlob) {
    if (!this.wsConnection || this.wsConnection.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected to backend");
    }

    // Convert blob to base64
    const base64 = await this.blobToBase64(imageBlob);

    const message = {
      type: "analyze_image",
      image: base64,
      timestamp: Date.now(),
    };

    this.wsConnection.send(JSON.stringify(message));
  }

  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
}

// Initialize the application when the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const app = new EcoSortAI();

  // Optional: Initialize backend integration
  // const backendIntegration = new PythonBackendIntegration(app);

  // Performance monitoring
  const performanceMonitor = new PerformanceMonitor();

  // Add to global scope for debugging
  window.ecoSortApp = app;
  window.performanceMonitor = performanceMonitor;

  console.log("Eco-Sort AI application initialized successfully!");
});
