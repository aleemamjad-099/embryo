# 🧬 Embryo Cell Detection & Health Analysis System

> **AI-powered embryo analysis using YOLOv8 object detection for IVF/ART clinical support**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Modules](#analysis-modules)
- [Results & Output](#results--output)
- [Development Stage Classification](#development-stage-classification)
- [Health Scoring Logic](#health-scoring-logic)
- [Sample Results](#sample-results)
- [Limitations & Future Work](#limitations--future-work)

---

## 🔬 Project Overview

This project applies **YOLOv8 object detection** to microscopic time-lapse images of human embryos captured during IVF (In Vitro Fertilization) procedures. The system automatically detects individual blastomeres (cells), counts them, estimates the developmental stage, and ranks cells by a composite health score.

The pipeline was trained on the **CleavageEmbryov1.1** dataset and evaluated across multiple embryo wells and time-points (e.g., `30.3 h`, `48.3 h`), mimicking real-world EmbryoScope time-lapse monitoring.

---

## 📊 Model Performance

The YOLOv8 model was trained for **50 epochs** on the CleavageEmbryov1.1 dataset. Key metrics at convergence:

| Metric | Value |
|---|---|
| **Precision** | ~0.95 |
| **Recall** | ~0.85 |
| **mAP@50** | ~0.91 |
| **mAP@50-95** | ~0.73 |
| **Inference Speed** | ~85–191 ms / image |
| **Input Size** | 640 × 640 px |

### Training Curves

All three loss components (box, classification, DFL) showed smooth, consistent convergence on both training and validation sets across 50 epochs, with no signs of overfitting.

### Confusion Matrix

| | Predicted: Cell | Predicted: Background |
|---|---|---|
| **True: Cell** | **2348** ✅ | 244 ❌ |
| **True: Background** | 382 ❌ | — |

- **True Positive Rate (Cell Recall):** `2348 / (2348 + 244)` ≈ **90.6%**
- **False Positive Rate:** 382 background regions misclassified as cells

> The model is conservative — it prefers higher precision over recall, reducing false cell detections in clinical contexts.

---

## 🗂️ Dataset

```yaml
# data.yaml
path: /content/CleavageEmbryov1.1/CleavageEmbryov1.1/
train: images/train
val:   images/val

names:
  0: cell
```

- **Single class:** `cell` (blastomere)
- **Source:** Time-lapse microscopy images from EmbryoScope-style incubators
- **Image format:** Grayscale `.jpg`, 800 × 800 px (circular well view)
- **Annotation format:** YOLO bounding boxes

---

## ✨ Features

- **Automatic cell detection** — counts individual blastomeres per frame
- **Developmental stage estimation** — maps cell count to embryo day
- **Cell health scoring** — composite score from detection confidence + morphological symmetry
- **Strongest cell identification** — highlights the best-quality blastomere with a green bounding box
- **Professional report overlay** — annotated image with stage, count, verdict, and timestamp
- **Auto-save reports** — timestamped output saved to local folders

---

## 📁 Project Structure

```
embryo-detection/
│
├── best.pt                         # Trained YOLOv8 weights
├── data.yaml                       # Dataset configuration
│
├── test_embryo.py                  # Basic inference script
├── heath_cell.ipynb                # Jupyter notebook — health analysis pipeline
├── Embryo.ipynb                    # (Additional analysis notebook)
│
├── images/
│   ├── 20230304-17097-3-F15-157.jpg            # Test image (13-cell embryo, ~30h)
│   └── D2024_06_02_S00699_I4778_P_WELL01_RUN248.jpg  # Test image (7-cell embryo, ~48h)
│
└── outputs/
    ├── confusion_matrix.png
    ├── results.png                 # Training curves
    ├── val_batch0_labels.jpg       # Ground truth labels
    ├── val_batch0_pred.jpg         # Model predictions
    └── final_report_output.jpg     # Sample annotated output
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/embryo-detection.git
cd embryo-detection

# Install dependencies
pip install ultralytics opencv-python numpy

# Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```

---

## 🚀 Usage

### Basic Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
results = model.predict(source='your_embryo_image.jpg', conf=0.5)

# Display result
im_array = results[0].plot()
cv2.imshow('Embryo Detection', im_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Full Analysis Pipeline (with Health Scoring)

```python
from ultralytics import YOLO
import cv2, datetime, os

model = YOLO('best.pt')
img_path = 'your_embryo_image.jpg'
img = cv2.imread(img_path)
results = model.predict(img_path, conf=0.5)
boxes = results[0].boxes

# Stage detection
count = len(boxes)
if count == 1:      stage = "Day 1 — Zygote"
elif count <= 4:    stage = "Day 2 — Early Cleavage"
elif count <= 8:    stage = "Day 3 — Morula Onset"
else:               stage = "Day 4+ — Advanced Stage"

# Health scoring
best_idx, best_score = -1, 0
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    w, h = x2 - x1, y2 - y1
    symmetry = min(w, h) / max(w, h)
    score = float(box.conf[0]) * 0.6 + symmetry * 0.4
    if score > best_score:
        best_score, best_idx = score, i

print(f"Stage: {stage} | Cells: {count} | Best Cell Score: {best_score:.2f}")
```

---

## 🧪 Analysis Modules

### Module 1 — Cell Strength Analyzer (`heath_cell.ipynb` — Cell 1)

Ranks all detected cells by a composite health score:

```
Health Score = (Confidence × 0.6) + (Symmetry × 0.4)
```

Outputs a ranked table and identifies the strongest blastomere.

---

### Module 2 — Embryo Health Report (`heath_cell.ipynb` — Cell 2)

Generates a full annotated image with:
- Green box → Most healthy cell
- Red boxes → All other cells
- Overlay: cell count, stage, timestamp
- Auto-saved to `Health_Analysis_Results/`

---

### Module 3 — Medical Report Overlay (`heath_cell.ipynb` — Cell 3)

Adds a professional HUD (Heads-Up Display) to the image:
- Total cell count
- Development stage
- Healthy cell probability
- Legend: "Green Box = Strongest"
- Auto-saved to `Embryo_Reports/`

---

### Module 4 — Professional Dashboard (`heath_cell.ipynb` — Cell 4)

Full clinical-style overlay with:
- Bordered dark panel header: `EMBRYO REPORT`
- Date & time stamp
- Verdict: `High Growth Potential` / `Slow Division` / `Monitor Further`
- Color-coded verdict (green / orange / white)
- Auto-saved to `Analysis_Results/`

---

## 📈 Development Stage Classification

| Cell Count | Estimated Stage | Label |
|---|---|---|
| 1 | Zygote | Day 1 |
| 2–4 | Early Cleavage | Day 2 |
| 5–8 | Morula Onset | Day 3 |
| 9+ | Advanced / Morula | Day 4+ |

---

## 🧮 Health Scoring Logic

The composite **Health Score** is calculated per detected cell:

```
Health Score = (YOLO Confidence × W₁) + (Aspect Ratio Symmetry × W₂)
```

| Component | Weight | Description |
|---|---|---|
| YOLO Confidence | 0.5–0.6 | Model's detection certainty |
| Symmetry (Aspect Ratio) | 0.4–0.5 | `min(w,h) / max(w,h)` → 1.0 = perfect circle |

> **Note:** Symmetry approximates morphological regularity. A perfect blastomere is spherical; deviations indicate fragmentation or irregular division.

**Example Output (13-cell embryo):**

```
Cell #1:  Confidence: 0.95,  Symmetry: 0.89,  Health Score: 0.92  ← STRONGEST
Cell #2:  Confidence: 0.92,  Symmetry: 0.80,  Health Score: 0.87
Cell #5:  Confidence: 0.86,  Symmetry: 0.37,  Health Score: 0.67  ← Most asymmetric
Cell #13: Confidence: 0.55,  Symmetry: 0.63,  Health Score: 0.59  ← Weakest
```

---

## 🖼️ Sample Results

### Test Image 1 — `20230304-17097-3-F15-157.jpg`
- **Time-point:** ~30.3 hours post-fertilization
- **Cells detected:** 13
- **Stage:** Day 4+ (Advanced Stage)
- **Verdict:** High Growth Potential (max confidence > 0.85)

### Test Image 2 — `D2024_06_02_S00699_I4778_P_WELL01_RUN248.jpg`
- **Time-point:** ~48.3 hours post-fertilization
- **Cells detected:** 7
- **Stage:** Day 3 (Morula Onset)
- **Healthy Cell Probability:** 0.89

### Validation Batch
- The model correctly detects cells across various well configurations (Well 1, Well 3, Well 5, Well 6)
- Handles single-cell (zygote) to multi-cell (morula) frames
- Confident detections (0.8–1.0) on clear, single embryos; lower confidence on fragmented/overlapping cells

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Overlapping bounding boxes:** NMS tuning may be needed for tightly packed morula-stage embryos
- **Single class only:** No distinction between normal blastomeres, fragments, or multinucleated cells
- **Symmetry proxy:** Aspect ratio is a coarse approximation of true morphological quality
- **No temporal tracking:** Each frame is analyzed independently; no cross-frame cell lineage

### Planned Improvements
- [ ] Add fragmentation detection as a separate class
- [ ] Implement NMS post-processing with IoU threshold tuning
- [ ] Integrate temporal tracking (DeepSORT or ByteTrack) for lineage analysis
- [ ] Add KIDScore / Gardner grading scale output
- [ ] Build a Gradio/Streamlit web interface for clinical use
- [ ] Export reports as PDF with patient metadata support

---

## 📚 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥ 8.0 | YOLOv8 model training & inference |
| `opencv-python` | ≥ 4.8 | Image loading, drawing, display |
| `numpy` | ≥ 1.24 | Array operations |
| `torch` | ≥ 2.0 | Deep learning backend |

---

## 📄 License

This project is intended for **research and educational purposes only**. It is **not** a certified medical device and must not be used as the sole basis for clinical decisions. Always consult qualified embryologists and clinicians for IVF procedures.

---

## 👤 Author

**Embryo AI Detection Project**
Built with ❤️ using YOLOv8 + OpenCV
Dataset: CleavageEmbryov1.1

---

> *"The goal is not to replace the embryologist — but to give them a second pair of AI-powered eyes."*
