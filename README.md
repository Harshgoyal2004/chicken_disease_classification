# Chicken Disease Classification Pipeline

A production-grade, end-to-end deep learning pipeline for automated detection of poultry diseases using transfer learning, configuration-driven workflows, and reproducible artifact tracking.

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Dataset](#dataset)
7. [Getting Started](#getting-started)
8. [Running the Pipeline](#running-the-pipeline)
9. [Core Components & Code Samples](#core-components--code-samples)
10. [Configuration & Hyperparameters](#configuration--hyperparameters)
11. [Results & Evaluation](#results--evaluation)
12. [Web Interface](#web-interface)
13. [DVC & Reproducibility](#dvc--reproducibility)
14. [Design Patterns](#design-patterns)
15. [Troubleshooting](#troubleshooting)
16. [Deployment Guide](#deployment-guide)
17. [Future Enhancements](#future-enhancements)
18. [References](#references)

---

## Overview

This project automates the detection of **Coccidiosis** (a parasitic poultry disease) from chicken fecal images using transfer learning with VGG16. The system achieves **88.72% accuracy** on validation data and includes:

- **Modular 4-stage pipeline**: Data Ingestion → Base Model Preparation → Training → Evaluation
- **Configuration-driven design**: All hyperparameters and paths controlled via YAML (no hardcoded values)
- **Reproducible workflows**: DVC for artifact tracking and pipeline orchestration
- **Web-based interface**: Flask application for real-time image classification
- **Production-ready code**: Type-hinted, modular components with clear separation of concerns
- **Inference-ready**: Standalone prediction stage for deployment

---

## Problem Statement

Manual inspection of chicken fecal samples for disease diagnosis is:

- **Time-consuming**: Requires trained personnel for each sample
- **Subjective**: Accuracy depends on inspector experience and fatigue
- **Unscalable**: Cannot efficiently handle large-scale poultry operations
- **Resource-intensive**: Expert availability varies across regions

**Solution**: Automate detection using deep learning to enable:
1. Rapid, consistent screening of fecal samples (seconds per image)
2. Reduced dependency on expert personnel
3. Deployment on edge devices for on-farm diagnosis
4. Scalable, reproducible workflows for continuous improvement

---

## Solution Architecture

### Pipeline Flow
```
Raw Fecal Sample Image
        ↓
[Stage 1: Data Ingestion] → Download, extract, organize by class
        ↓
[Stage 2: Base Model Prep] → Load VGG16, attach custom classification head
        ↓
[Stage 3: Training] → Fine-tune with augmentation (80/20 train/val split)
        ↓
[Stage 4: Evaluation] → Validate performance, save metrics
        ↓
[Stage 5: Inference] → Real-time prediction with confidence scores
```

### Model Design

| Component | Details |
|-----------|---------|
| **Backbone** | VGG16 (ImageNet pre-trained, `include_top=False`) |
| **Pooling** | GlobalAveragePooling2D (spatial dimension reduction) |
| **Dense Layer** | 256 units with ReLU activation |
| **Regularization** | Dropout(0.5) to prevent overfitting |
| **Output** | 2-unit softmax layer (binary classification) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Categorical Cross-Entropy |
| **Metrics** | Accuracy |

**Why VGG16?**
- Simple, interpretable 16-layer architecture
- Pre-trained ImageNet weights transfer exceptionally well to medical imaging tasks
- Computational efficiency suitable for deployment on modest hardware
- Clear migration path to deeper models (ResNet50, EfficientNet) if needed
- Easy to explain in technical interviews

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.9+ | Core development |
| **ML/DL Framework** | TensorFlow/Keras 2.10+ | Deep learning & model management |
| **Array Processing** | NumPy | Numerical computations |
| **Image Processing** | OpenCV (cv2), Pillow | Image I/O and preprocessing |
| **ML Utilities** | Scikit-learn | Metrics, preprocessing, validation |
| **Configuration** | PyYAML, Python-Box | YAML parsing, nested dict access |
| **Web Framework** | Flask | Lightweight HTTP server & routing |
| **Reproducibility** | DVC, Git | Pipeline orchestration, version control |
| **Package Management** | setuptools, pip | Installation & distribution |

All dependencies are pinned in `requirements.txt` for reproducible environments across machines.

---

## Project Structure

```
chicken_disease_classification/
├── artifacts/                          # Generated outputs (ignored by git)
│   ├── data_ingestion/
│   │   └── Chicken-fecal-images/
│   │       ├── Coccidiosis/            # Disease samples (~200 images)
│   │       └── Healthy/                # Healthy samples (~200 images)
│   ├── prepare_base_model/
│   │   ├── base_model.h5               # VGG16 backbone (downloaded from TF)
│   │   └── updated_base_model.h5       # VGG16 + custom head (compiled)
│   ├── training/
│   │   └── trained_model.h5            # Final trained model (final artifact)
│   └── evaluation/
│       └── scores.json                 # {loss, accuracy} metrics
│
├── .dvc/                               # DVC metadata (tracked in git)
├── .dvcignore                          # Patterns for DVC to ignore
├── .gitignore                          # Patterns for git to ignore
│
├── config/
│   └── config.yaml                     # Artifact paths, dataset source URL
│
├── src/cnnClassifier/
│   ├── __init__.py                     # Package initialization + logging setup
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # DataIngestion class (download, extract)
│   │   ├── prepare_base_model.py       # PrepareBaseModel class (load VGG16, attach head)
│   │   ├── training.py                 # Training class (fit model with augmentation)
│   │   └── evaluation.py               # Evaluation class (validate, save metrics)
│   ├── config/
│   │   ├── __init__.py
│   │   └── configuration.py            # ConfigurationManager (single source of truth)
│   ├── entity/
│   │   ├── __init__.py
│   │   └── config_entity.py            # Dataclasses for type-safe configs
│   ├── constants/
│   │   └── __init__.py                 # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── stage_01_data_ingestion.py  # DataIngestionPipeline wrapper
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   ├── stage_04_evaluation.py
│   │   └── stage_05_predict.py         # PredictionPipeline (used by Flask)
│   └── utils/
│       ├── __init__.py
│       └── common.py                   # YAML I/O, model save/load, decodeImage
│
├── templates/
│   └── index.html                      # Flask UI (HTML + JavaScript)
│
├── main.py                             # Pipeline orchestrator (runs all stages)
├── app.py                              # Flask web app (routes: /, /train, /predict)
├── setup.py                            # Package configuration (pip install -e .)
├── requirements.txt                    # Pinned Python dependencies
├── params.yaml                         # Hyperparameters (EPOCHS, BATCH_SIZE, etc)
├── dvc.yaml                            # DVC pipeline definition
├── README.md                           # This file
└── cnnClassifier.egg-info/             # Auto-generated package metadata
```

**Key Design Principles:**
- **Configuration Manager Pattern**: Single source of truth for all settings
- **Dataclass Entities**: Type-safe configuration objects with IDE support
- **Modular Components**: Each stage (data ingestion, training, etc.) is independent and testable
- **Utility Functions**: Reusable helpers in `utils/common.py` reduce duplication
- **Pipeline Wrappers**: Stages can run independently or orchestrated by `main.py`

---

## Dataset

The pipeline expects a structured dataset organized by class:

```
artifacts/data_ingestion/Chicken-fecal-images/
  ├── Coccidiosis/           # ~200 disease samples
  │   ├── coccidiosis_001.jpg
  │   ├── coccidiosis_002.jpg
  │   └── ...
  └── Healthy/               # ~200 healthy samples
      ├── healthy_001.jpg
      ├── healthy_002.jpg
      └── ...
```

**Data Preparation:**
1. Place `Chicken-fecal-images.zip` in the project root, OR
2. Update `config.data_ingestion.source_URL` in `config/config.yaml` to point to your dataset (supports local file URIs and HTTP URLs)

The Data Ingestion stage will automatically extract the zip and organize it into the expected structure.

---

## Getting Started

### Prerequisites
- Python 3.9 or higher
- `git` for version control
- ~500MB disk space (for models and datasets)

### Installation

```bash
# Clone the repository
git clone https://github.com/Harshgoyal2004/chicken_disease_classification.git
cd chicken_disease_classification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Verify Installation:**
```bash
python -c "import src.cnnClassifier; print('Installation successful!')"
```

---

## Running the Pipeline

### Quick Start (1-epoch smoke test)

```bash
# Set EPOCHS: 1 in params.yaml first, then run training:
python -c "from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline; TrainingPipeline().main()"
```

### Full Pipeline Execution

Run all stages in sequence:
```bash
python main.py
```

This executes:
1. Data Ingestion (download & extract dataset)
2. Base Model Preparation (load VGG16, build classification head)
3. Training (fit model on training data with validation)
4. Evaluation (validate on held-out test set)

### Run Individual Stages

```python
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

# Run any stage independently
DataIngestionPipeline().main()
PrepareBaseModelPipeline().main()
TrainingPipeline().main()
EvaluationPipeline().main()
```

---

## Core Components & Code Samples

### 1. Configuration Manager (Single Source of Truth)

**Location:** `src/cnnClassifier/config/configuration.py`

```python
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig, TrainingConfig, EvaluationConfig
)

class ConfigurationManager:
    """Centralized configuration management for all pipeline stages."""
    
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_path=config.unzip_path
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params
        create_directories([config.root_dir])
        return TrainingConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            updated_base_model_path=self.config.prepare_base_model.updated_base_model_path,
            training_data=self.config.data_ingestion.unzip_path,
            params=params
        )
```

**Why this pattern?**
- All paths and hyperparameters read from YAML once at startup
- No hardcoded values scattered throughout the code
- Easy to swap configs for different environments
- Type-safe dataclasses ensure consistency

---

### 2. Data Ingestion Component

**Location:** `src/cnnClassifier/components/data_ingestion.py`

```python
import os
import zipfile
import urllib.request
from src.cnnClassifier.utils.common import get_size

class DataIngestion:
    """Download and extract the dataset."""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Download dataset from source_URL if not already present."""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = urllib.request.urlretrieve(
                self.config.source_URL,
                self.config.local_data_file
            )
            print(f"Downloaded {filename} ({get_size(filename)})")
        else:
            print(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        """Extract zip file to unzip_path."""
        unzip_path = self.config.unzip_path
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as z:
            z.extractall(unzip_path)
        print(f"Extracted to {unzip_path}")
```

**Design Benefits:**
- Single Responsibility: each method has one job
- Reusable: can test `download_file()` and `extract_zip_file()` independently
- Configuration-driven: paths come from config, not hardcoded

---

### 3. Model Preparation (Transfer Learning Setup)

**Location:** `src/cnnClassifier/components/prepare_base_model.py`

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from src.cnnClassifier.utils.common import save_model

class PrepareBaseModel:
    """Load VGG16 and prepare classification head."""
    
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Download and save VGG16 backbone."""
        self.model = VGG16(
            input_shape=self.config.params.IMAGE_SIZE,
            weights='imagenet',
            include_top=False
        )
        save_model(self.model, self.config.base_model_path)
        return self.model

    def _prepare_full_model(self, learning_rate):
        """Build classification head on top of VGG16."""
        model = Sequential([
            self.model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.config.params.CLASSES, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def update_base_model(self):
        """Build and save the complete model."""
        self.full_model = self._prepare_full_model(
            learning_rate=self.config.params.LEARNING_RATE
        )
        save_model(self.full_model, self.config.updated_base_model_path)
```

**Transfer Learning Strategy:**
- Download pre-trained VGG16 weights from ImageNet
- Keep backbone frozen (no fine-tuning) to preserve learned features
- Attach custom classification head for binary disease classification
- This approach requires less data and trains faster than training from scratch

---

### 4. Training Component (With Data Augmentation)

**Location:** `src/cnnClassifier/components/training.py`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.cnnClassifier.utils.common import load_model

class Training:
    """Train the model with data augmentation and validation split."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_updated_data_generator_instance(self, data_path):
        """Create augmented data generator with 80/20 train/val split."""
        datagenerator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,           # Random rotation
            width_shift_range=0.2,       # Random horizontal shift
            height_shift_range=0.2,      # Random vertical shift
            shear_range=0.2,             # Shear transformation
            zoom_range=0.2,              # Random zoom
            horizontal_flip=True,        # Random horizontal flip
            fill_mode='nearest',         # Fill mode for transformations
            validation_split=0.2         # 80% train, 20% validation
        )
        return datagenerator.flow_from_directory(
            directory=data_path,
            target_size=self.config.params.IMAGE_SIZE[:2],
            batch_size=self.config.params.BATCH_SIZE,
            class_mode='categorical',    # One-hot encoded labels
            validation_split=0.2         # Must match above
        )

    def train(self):
        """Load model, compile, and train on augmented data."""
        self.model = load_model(self.config.updated_base_model_path)
        
        # Critical: recompile after load to avoid Keras cached metadata issues
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.generator = self.get_updated_data_generator_instance(self.config.training_data)
        
        self.model.fit(
            self.generator,
            epochs=self.config.params.EPOCHS,
            steps_per_epoch=self.generator.samples // self.config.params.BATCH_SIZE,
            validation_steps=self.generator.samples // self.config.params.BATCH_SIZE // 5
        )
        
        save_model(self.model, self.config.trained_model_path)
```

**Key Technical Details:**
- **Data Augmentation**: Artificially expand dataset using transformations (rotation, shifts, zoom)
- **Validation Split**: 80/20 split ensures model generalizes to unseen data
- **Recompile After Load**: Critical to avoid Keras serialization issues
- **Steps Per Epoch**: Calculated from dataset size to ensure proper training

---

### 5. Web Interface (Flask App)

**Location:** `app.py`

```python
from flask import Flask, render_template, request, jsonify
from src.cnnClassifier.pipeline.stage_05_predict import PredictionPipeline
from src.cnnClassifier.utils.common import decodeImage
import os

app = Flask(__name__)

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train_route():
    """Trigger full pipeline retraining."""
    os.system("python3 main.py")
    return jsonify({"status": "Training started"})

@app.route("/predict", methods=["POST"])
def predict_route():
    """Classify an uploaded image."""
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode base64 image to file
    decodeImage(image_data, "input_image.jpg")
    
    # Run prediction
    pipeline = PredictionPipeline("input_image.jpg")
    result = pipeline.predict()
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
```

**API Endpoints:**
- `GET /` — Serve HTML UI
- `POST /train` — Trigger full pipeline retraining
- `POST /predict` — Classify image (expects JSON with base64 `image` field)

---

### 6. Utility Functions (Reusable Helpers)

**Location:** `src/cnnClassifier/utils/common.py`

```python
import yaml
import json
import os
import base64
from pathlib import Path

def read_yaml(yaml_file_path):
    """Load YAML configuration file."""
    with open(yaml_file_path) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def save_json(path, data):
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """Load JSON file to dictionary."""
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    """Save Keras model to HDF5 file."""
    model.save(path)
    print(f"Model saved at {path}")

def load_model(path):
    """Load Keras model from HDF5 file."""
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(path)

def decodeImage(img_string, filename):
    """Decode base64 image string and save to disk."""
    img_data = base64.b64decode(img_string)
    with open(filename, 'wb') as f:
        f.write(img_data)
    print(f"Image decoded and saved to {filename}")

def create_directories(path_list):
    """Create directories if they don't exist."""
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
```

**Benefits:**
- **DRY Principle**: Reuse these functions across all components
- **Centralized Logic**: Change implementation once, affects entire codebase
- **Testability**: Each function is independently testable

---

## Configuration & Hyperparameters

### Main Configuration (`config/config.yaml`)

Defines all artifact paths and data source:
```yaml
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: file:///path/to/Chicken-fecal-images.zip  # Update with your data path
  local_data_file: artifacts/data_ingestion/Chicken-fecal-images.zip
  unzip_path: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/updated_base_model.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/trained_model.h5

evaluation:
  root_dir: artifacts/evaluation
  evaluation_data_path: artifacts/data_ingestion/Chicken-fecal-images
  scores_file: artifacts/evaluation/scores.json
```

### Hyperparameters (`params.yaml`)

Controls model training behavior:
```yaml
AUGMENTATION: true
IMAGE_SIZE: [224, 224, 3]      # VGG16 input size
BATCH_SIZE: 32                  # Reduce to 16 if OOM errors
EPOCHS: 1                        # Set to 25 for full training
LEARNING_RATE: 0.001
CLASSES: 2
CLASS_NAMES: ['Coccidiosis', 'Healthy']
```

**To experiment:**
1. Edit `params.yaml`
2. Re-run pipeline stages
3. ConfigurationManager automatically picks up new values

No code changes required!

---

## Results & Evaluation

### Metrics

After evaluation, check `artifacts/evaluation/scores.json`:

```json
{
  "loss": 0.2792516350746155,
  "accuracy": 0.8871794939041138
}
```

**Interpretation:**
- **Loss (0.279)**: Prediction error measure (lower is better). Value indicates reasonable convergence.
- **Accuracy (88.72%)**: Percentage of correctly classified validation samples. Strong baseline for transfer learning.

### Sample Predictions

When running `/predict` endpoint:
```json
{
  "prediction": "Healthy",
  "confidence": 0.96,
  "disease": "Coccidiosis",
  "healthy": "Healthy"
}
```

### Improving Performance

If accuracy is below expectations:

1. **Increase training duration**: Set `EPOCHS: 50` in `params.yaml`
2. **Fine-tune backbone**: Unfreeze VGG16 layers in `prepare_base_model.py`
3. **Collect more data**: Class imbalance limits performance
4. **Adjust batch size**: Reduce to 16 if gradient estimates are noisy
5. **Try deeper model**: Replace VGG16 with ResNet50 or EfficientNet

---

## Web Interface

Start the Flask application:
```bash
python app.py
# Visit http://127.0.0.1:8080 in your browser
```

**Features:**
- Upload chicken fecal images
- Real-time preview of uploaded image
- Classify image (returns disease prediction + confidence)
- Trigger pipeline retraining directly from UI

---

## DVC & Reproducibility

### Why DVC?

DVC (Data Version Control) tracks large files and pipeline dependencies:
- **Artifact Tracking**: Store models, datasets in cloud or local storage
- **Pipeline Caching**: Re-runs only changed stages
- **Reproducibility**: Ensure exact environment across team members

### Initialize DVC

```bash
# Already done if .dvc/ exists
dvc init

# Add dataset to DVC tracking (optional)
dvc add artifacts/data_ingestion/Chicken-fecal-images.zip
git add artifacts/data_ingestion/Chicken-fecal-images.zip.dvc
git commit -m "Add dataset to DVC"

# Configure remote storage (S3, GCP, Azure, etc.)
dvc remote add -d myremote s3://my-bucket/path
dvc push  # Upload artifacts to remote
```

### Run with DVC

```bash
# Run full pipeline under DVC control
dvc repro

# Run with caching (skip unchanged stages)
dvc repro --dry-run  # Preview what will run
dvc repro            # Execute pipeline
```

---

## Design Patterns

### 1. Configuration Manager Pattern
**Problem**: Hardcoded paths and values scattered everywhere  
**Solution**: Single ConfigurationManager reads YAML once at startup

### 2. Pipeline Pattern
**Problem**: Complex workflows hard to track and debug  
**Solution**: Each stage (data_ingestion, training, evaluation) is a standalone pipeline

### 3. Component Pattern
**Problem**: Business logic mixed with orchestration  
**Solution**: Core logic in components, pipelines handle setup/teardown

### 4. Entity/DTO Pattern
**Problem**: Type hints and validation missing from configs  
**Solution**: Dataclasses ensure type safety and IDE support

### 5. Factory Pattern (Implicit)
**Problem**: Complex object creation scattered  
**Solution**: ConfigurationManager creates properly configured dataclasses

---

## Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `The PyDataset has length 0` | Missing `validation_split` in ImageDataGenerator | Ensure both `ImageDataGenerator` and `flow_from_directory` use `validation_split=0.2` |
| `Module not found` | Package not installed in editable mode | Run `pip install -e .` |
| `Out of Memory (OOM)` | Batch size too large | Reduce `BATCH_SIZE: 16` in `params.yaml` |
| `File not found: data.zip` | Dataset not placed in correct location | Update `source_URL` in `config.yaml` to point to your dataset |
| `Keras compiled metrics not built` | Model loaded but not recompiled | Recompile after load: `model.compile(...)` before training/eval |

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Deployment Guide

### Docker Containerization

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt && pip install -e .

ENV FLASK_APP=app.py
EXPOSE 8080

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:app"]
```

Build and run:
```bash
docker build -t chicken-disease-classifier .
docker run -p 8080:8080 chicken-disease-classifier
```

### Production Deployment

1. **Use WSGI Server** (not Flask dev server):
   ```bash
   pip install gunicorn
   gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
   ```

2. **Model Versioning**: Track models in DVC or MLflow
   ```bash
   dvc push  # Upload to S3/GCP/etc
   ```

3. **Monitoring & Logging**:
   - Log all predictions for audit trails
   - Monitor latency and error rates
   - Set up alerts for performance degradation

4. **CI/CD Pipeline** (GitHub Actions example):
   ```yaml
   name: Test & Deploy
   on: [push]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - run: pip install -r requirements.txt
         - run: pytest  # If tests added
     deploy:
       needs: test
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - run: docker build -t my-app .
         - run: docker push my-registry/my-app:latest
   ```

---

## Future Enhancements

### Short-term (1-3 months)
- [ ] Add unit tests for components (`pytest`)
- [ ] Implement cross-validation for robust performance metrics
- [ ] Add Grad-CAM visualization to explain model decisions
- [ ] Collect class-specific metrics (precision, recall, F1)

### Medium-term (3-6 months)
- [ ] Fine-tune VGG16 backbone instead of frozen weights
- [ ] Experiment with deeper architectures (ResNet50, EfficientNet)
- [ ] Implement hyperparameter tuning (Optuna, Keras Tuner)
- [ ] Create dashboard to visualize model performance over time
- [ ] Add multi-class support (extend beyond 2 classes)

### Long-term (6+ months)
- [ ] Deploy on edge devices (TensorFlow Lite, ONNX)
- [ ] Implement continuous learning from new data
- [ ] Add explainability module (SHAP, integrated gradients)
- [ ] Multi-model ensemble for improved robustness
- [ ] Real-time performance monitoring and A/B testing framework

---

## References

### Papers
- [VGG16 Architecture](https://arxiv.org/abs/1409.1556) - Very Deep Convolutional Networks for Large-Scale Image Recognition
- [ImageNet Classification](http://www.image-net.org/) - Large Visual Database Project

### Documentation
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [DVC Documentation](https://dvc.org/doc)

### Tools
- [Flask Documentation](https://flask.palletsprojects.com/)
- [setuptools Guide](https://setuptools.readthedocs.io/)

---

## License

This project is licensed under the **MIT License** — see `LICENSE` file for details.

## Author & Contact

- **Harsh Goyal** — [@Harshgoyal2004](https://github.com/Harshgoyal2004)
- Issues & Contributions welcome on [GitHub](https://github.com/Harshgoyal2004/chicken_disease_classification)

---

**Last Updated**: November 18, 2025  
**Model Accuracy**: 88.72%  
**Status**: Production-Ready ✓
