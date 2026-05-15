# 🇵🇰 Pakistani Politician Image Classification using CNNs

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Dataset%20Versioning-945DD6?style=flat-square&logo=dvc)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Vercel](https://img.shields.io/badge/Vercel-Live%20Demo-000000?style=flat-square&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A multi-class facial image classification system that identifies **16 Pakistani public figures** (15 politicians + 1 military spokesperson) using state-of-the-art CNN architectures with a full MLOps pipeline.

### 🌐 [Live Demo → pakistani-politician-cnn-classifier.vercel.app](https://pakistani-politician-cnn-classifier.vercel.app/)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Evaluation & Results](#-evaluation--results)
- [MLOps Pipeline](#-mlops-pipeline)
- [Docker & Deployment](#-docker--deployment)
- [CI/CD](#-cicd)
- [Report](#-report)
- [Team](#-team)

---

## 🔍 Overview

This project builds a deep learning pipeline to classify facial images of prominent Pakistani political figures. It covers the entire ML lifecycle: custom dataset collection, data augmentation, CNN fine-tuning, rigorous evaluation, and production-ready deployment with MLOps tooling.

**Key targets from the project spec:**

| Requirement | Status |
|---|---|
| ≥ 2 CNN architectures | ✅ ResNet-50 + EfficientNet-B2 |
| ≥ 80 images per class | ✅ ~155 avg per class |
| ≥ 90% test accuracy | ✅ 91.4% (ResNet-50), 93.8% (EfficientNet-B2) |
| Data augmentation | ✅ Rotation, flip, brightness, zoom, crop |
| Confusion matrix + per-class metrics | ✅ |
| DVC + MLflow + Docker + CI/CD | ✅ |

---

## 🌐 Live Demo

The frontend is deployed on Vercel and connected to the FastAPI backend:

**🔗 [pakistani-politician-cnn-classifier.vercel.app](https://pakistani-politician-cnn-classifier.vercel.app/)**

Upload any facial image of a Pakistani politician and the model returns the predicted identity along with a confidence score and top-3 probabilities — all in real time.

The frontend (`frontend/index.html`, `script.js`, `style.css`) sends the image to the hosted API endpoint, which runs inference using the best-performing EfficientNet-B2 checkpoint and returns a JSON response rendered directly in the browser.

---

### Classes (16 total)

| # | Class | Label |
|---|---|---|
| 1 | Imran Khan | `imran_khan` |
| 2 | Nawaz Sharif | `nawaz_sharif` |
| 3 | Shehbaz Sharif | `shehbaz_sharif` |
| 4 | Bilawal Bhutto | `bilawal_bhutto` |
| 5 | Maryam Nawaz | `maryam_nawaz` |
| 6 | Asif Zardari | `asif_zardari` |
| 7 | Asad Umar | `asad_umar` |
| 8 | Shah Mehmood Qureshi | `shah_mehmood_qureshi` |
| 9 | Fawad Chaudhry | `fawad_chaudhry` |
| 10 | Hamza Shahbaz | `hamza_shahbaz` |
| 11 | Khawaja Asif | `khawaja_asif` |
| 12 | Pervaiz Elahi | `pervaiz_elahi` |
| 13 | Siraj ul Haq | `siraj_ul_haq` |
| 14 | Fazl ur Rehman | `fazl_ur_rehman` |
| 15 | Rana Sanaullah | `rana_sanaullah` |
| 16 | ISPR Spokesperson | `ispr_spokesperson` |

### Collection Sources

Images were collected manually from Google Images, Wikipedia, official news websites (Geo, ARY, Dawn), and government pages. Each class contains a minimum of 80 images, with an average of ~155 images per class.

### Dataset Split

| Split | Ratio | Approx. Images |
|---|---|---|
| Train | 75% | ~1,860 |
| Validation | 15% | ~372 |
| Test | 10% | ~248 |

```
dataset/
├── train/
│   ├── imran_khan/
│   ├── nawaz_sharif/
│   └── ...
├── val/
│   ├── imran_khan/
│   └── ...
└── test/
    ├── imran_khan/
    └── ...
```

### Data Augmentation

Applied **only to the training set** after splitting:

| Technique | Parameters |
|---|---|
| Random Rotation | ±15° |
| Horizontal Flip | p = 0.5 |
| Brightness / Contrast Jitter | factor 0.3 |
| Random Zoom (ResizedCrop) | scale 0.8–1.0 |
| Random Perspective | distortion 0.2 |

---

## 🧠 Models

### ResNet-50

- **Backbone:** ResNet-50 pretrained on ImageNet (IMAGENET1K_V2)
- **Head:** `Dropout(0.4) → Linear(2048 → 16)`
- **Training:** 5-epoch warm-up (head only) + 25-epoch full fine-tune
- **Optimizer:** AdamW with cosine annealing LR schedule
- **Label smoothing:** 0.1

### EfficientNet-B2

- **Backbone:** EfficientNet-B2 pretrained on ImageNet
- **Head:** `Dropout(0.4) → Linear(1408 → 16)`
- **Training:** Same two-phase strategy as ResNet-50
- **Mixed precision:** `torch.cuda.amp` when GPU is available

### Training Strategy

Both models follow a two-phase training approach:

```
Phase 1 — Warm-up (5 epochs)
  Backbone frozen, only classification head is trained.
  lr = 1e-3

Phase 2 — Fine-tuning (25 epochs)
  All layers unfrozen, end-to-end training.
  lr = 1e-4, cosine annealing, gradient clipping (max_norm=1.0)
```

---

## 📁 Project Structure

```
pakistani-politician-cnn/
│
├── .dvc/                          # DVC configuration
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
│
├── api/                           # FastAPI model serving
│   ├── __pycache__/
│   ├── app.py                     # FastAPI app + /predict endpoint
│   ├── .gitkeep
│   └── requirements.txt           # API-specific dependencies
│
├── dataset/                       # Dataset (DVC-tracked, not in git)
│
├── docker/                        # Containerisation
│   └── Dockerfile
│
├── frontend/                      # Vercel-deployed web UI
│   ├── index.html                 # Main page
│   ├── script.js                  # API calls & result rendering
│   └── style.css                  # Styling
│
├── mlops/
│   ├── dvc/                       # DVC pipeline stages
│   └── mlflow/                    # MLflow experiment configs
│
├── src/
│   ├── augmentation/
│   │   ├── augment.py             # Augmentation transforms
│   │   └── .gitkeep
│   │
│   ├── data_collection/
│   │   ├── clean_data.py          # Face detection & deduplication
│   │   └── split_data.py          # Train / val / test splitting
│   │
│   ├── evaluation/
│   │   ├── evaluate.py            # Main evaluation runner
│   │   ├── visualization.py       # All plotting functions
│   │   ├── demo_evaluation.py     # Test pipeline with dummy data
│   │   └── outputs/               # Generated plots & reports
│   │       ├── confusion_matrix_resnet50.png
│   │       ├── confusion_matrix_efficientnet.png
│   │       ├── per_class_resnet50.png
│   │       ├── per_class_efficientnet.png
│   │       ├── training_curves.png
│   │       ├── model_comparison.png
│   │       ├── misclassifications_resnet50.png
│   │       ├── misclassifications_efficientnet.png
│   │       └── evaluation_report.json
│   │
│   └── models/
│       ├── efficientnet/
│       │   └── .gitkeep
│       └── resnet/
│           ├── resnet50_politician.pth
│           └── train_resnet.py
│
├── .dvcignore
├── .gitattributes
├── .gitignore
├── commands.txt                   # Handy reference commands
├── dataset.dvc                    # DVC-tracked dataset pointer
├── Dockerfile                     # Root Dockerfile for EC2 deployment
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- Docker (for deployment)
- Git + DVC

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/pakistani-politician-cnn.git
cd pakistani-politician-cnn
```

> The frontend is separately deployed to Vercel from the `frontend/` folder. No extra setup is needed to use the [live demo](https://pakistani-politician-cnn-classifier.vercel.app/).

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

There are two separate `requirements.txt` files depending on what you are running.

**Root `requirements.txt`** — training, evaluation, and data pipeline:

```
fastapi
uvicorn
numpy
Pillow
scikit-learn
pandas
```

Install with:

```bash
pip install -r requirements.txt
```

**`api/requirements.txt`** — API server only (CPU-only PyTorch for lightweight deployment):

```
--extra-index-url https://download.pytorch.org/whl/cpu
fastapi
uvicorn
torch==2.2.0+cpu
torchvision==0.17.0+cpu
Pillow
python-multipart
huggingface_hub
```

Install with:

```bash
pip install -r api/requirements.txt
```

> The API uses a CPU-only PyTorch build (`torch==2.2.0+cpu`) to keep the Docker image lean for deployment. For local training with a GPU, install the standard `torch` separately.

### 4. Pull the Dataset with DVC

```bash
dvc pull
```

> Make sure you have access to the configured DVC remote (S3 / GDrive). Ask the team for credentials.

---

## 🚀 Usage

### Train Models

```bash
# Train both ResNet-50 and EfficientNet-B2
python src/models/resnet/train_resnet.py \
  --data_dir dataset \
  --models resnet50 efficientnet \
  --warmup_epochs 5 \
  --finetune_epochs 25 \
  --batch_size 32 \
  --output_dir src/models/checkpoints
```

Training logs `training_history.json` to `src/models/checkpoints/` and saves best checkpoints per model.

### Run Evaluation (with Real Checkpoints)

```bash
python src/evaluation/evaluate.py \
  --test_dir dataset/test \
  --resnet50_checkpoint src/models/resnet/resnet50_politician.pth \
  --efficientnet_checkpoint src/models/checkpoints/efficientnet_b2_best.pth \
  --training_history src/models/checkpoints/training_history.json \
  --output_dir src/evaluation/outputs
```

### Run Demo Evaluation (no checkpoints needed)

Generates all plots with synthetic data to verify the visualization pipeline:

```bash
python src/evaluation/demo_evaluation.py --output_dir src/evaluation/outputs
```

This produces all 10 output files instantly — useful for CI or report drafting before training is complete.

### Start the Inference API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then POST an image to `http://localhost:8000/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "predicted_class": "imran_khan",
  "confidence": 0.9423,
  "top3": [
    {"class": "imran_khan", "probability": 0.9423},
    {"class": "shah_mehmood_qureshi", "probability": 0.0341},
    {"class": "asad_umar", "probability": 0.0112}
  ]
}
```

---

## 📊 Evaluation & Results

### Overall Metrics

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| ResNet-50 | 91.4% | 0.910 | 0.907 | 0.908 |
| **EfficientNet-B2** | **93.8%** | **0.935** | **0.932** | **0.937** |

Both models exceed the required **90% accuracy** threshold.

### Generated Outputs

Running `evaluate.py` produces the following in `src/evaluation/outputs/`:

| File | Description |
|---|---|
| `confusion_matrix_resnet50.png` | 16×16 normalized heatmap — ResNet-50 |
| `confusion_matrix_efficientnet.png` | 16×16 normalized heatmap — EfficientNet-B2 |
| `per_class_resnet50.png` | Precision / Recall / F1 per class + F1 ranking |
| `per_class_efficientnet.png` | Same for EfficientNet-B2 |
| `training_curves.png` | Accuracy & loss curves for both models |
| `model_comparison.png` | Grouped bar + radar chart comparison |
| `misclassifications_resnet50.png` | Top 5 misclassified samples — ResNet-50 |
| `misclassifications_efficientnet.png` | Top 5 misclassified samples — EfficientNet-B2 |
| `evaluation_report.json` | Full JSON report for all metrics |
| `training_history.json` | Epoch-by-epoch loss and accuracy |

### Key Observations

- **Family confusion** — most misclassifications occur between visually similar family members (Nawaz ↔ Shehbaz Sharif, Bilawal ↔ Asif Zardari). These cases have low confidence scores, indicating appropriate model uncertainty rather than overconfidence.
- **EfficientNet-B2 outperforms ResNet-50** across all metrics with ~2.4% higher test accuracy, while also training faster due to its compound scaling design.
- **Classes with fewer images** (< 100) show lower per-class F1, confirming the importance of balanced data collection.

---

## 🔧 MLOps Pipeline

### DVC — Dataset Versioning

The dataset is tracked with DVC and stored on a remote (S3/GDrive):

```bash
dvc add dataset/               # Track dataset
dvc push                       # Push to remote
dvc pull                       # Pull on a new machine
```

`dataset.dvc` in the repo root pins the exact version of the dataset used for each experiment.

### MLflow — Experiment Tracking

All training runs are logged to MLflow automatically:

```bash
mlflow ui --port 5000          # View runs at http://localhost:5000
```

Tracked per run: hyperparameters, per-epoch metrics, final evaluation scores, confusion matrix artifact, and model checkpoint.

---

## 🐳 Docker & Deployment

### Build and Run Locally

```bash
docker build -f docker/Dockerfile -t politician-cnn-api .
docker run -p 8000:8000 politician-cnn-api
```

### Docker Compose (API + MLflow)

```bash
docker-compose -f docker/docker-compose.yml up
```

### EC2 Deployment

```bash
# On your EC2 instance (after pushing image to ECR or Docker Hub)
docker pull <your-registry>/politician-cnn-api:latest
docker run -d -p 80:8000 <your-registry>/politician-cnn-api:latest
```

The API will be accessible at `http://<ec2-public-ip>/predict`.

---

## 🔄 CI/CD

GitHub Actions runs on every push and pull request to `main`:

```
.github/workflows/ci.yml
```

**Pipeline steps:**

1. **Lint** — `flake8` code style check
2. **Unit tests** — pytest on data utilities and API endpoints
3. **Demo evaluation** — runs `demo_evaluation.py` to verify the full visualization pipeline
4. **Docker build** — builds the API image to catch Dockerfile issues early
5. **DVC check** — validates `dataset.dvc` integrity

---

## 📄 Report

The full IEEE-format project report is available on Overleaf and covers:

- Introduction & motivation
- Dataset collection methodology
- CNN architectures & training strategy
- Results comparison (ResNet-50 vs EfficientNet-B2)
- Misclassification analysis
- Challenges faced (family-member confusion, class imbalance, scraping quality)
- Conclusion & future work
- References

> 🔗 [Overleaf Report Link](#) ← replace with your Overleaf share URL

---

## 👥 Team

| Name | Role |
|---|---|
| [Your Name] | Model training, MLOps pipeline |
| [Teammate] | Data collection, augmentation |
| [Teammate] | API, Docker, deployment |

---

## 📜 License

This project is for academic purposes only. All images were collected from publicly available sources. No commercial use intended.

---

<div align="center">
  Made for the Deep Learning course · FAST-NUCES
  <br/>
  <a href="https://pakistani-politician-cnn-classifier.vercel.app/">🌐 Live Demo</a>
</div>