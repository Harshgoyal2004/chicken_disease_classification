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
17. [CI/CD (GitHub Actions)](#ci-cd-github-actions)
18. [Logging](#logging)
19. [API Response Format](#api-response-format)
20. [DVC Pipeline Stages](#dvc-pipeline-stages)
21. [Future Enhancements](#future-enhancements)
22. [References](#references)
23. [License](#license)
24. [Author & Contact](#author--contact)

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
[Stage 5: Inference] → Real-time prediction
```

### Model Design

| Component | Details |
|-----------|---------|
| **Backbone** | VGG16 (ImageNet pre-trained, `include_top=False`) |
| **Pooling** | GlobalAveragePooling2D |
| **Head** | Dense(2, activation="softmax") |
| **Backbone Trainability** | Frozen (`freeze_all=True`) |
| **Optimizer** | Adam (lr from `params.yaml`, default 0.001) |
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
| **Language** | Python 3.9+ |
| **ML/DL Framework** | TensorFlow/Keras 2.10+ |
| **Array Processing** | NumPy |
| **ML Utilities** | Scikit-learn |
| **Configuration** | PyYAML, python-box |
| **Web Framework** | Flask, Flask-Cors |
| **Reproducibility** | DVC, Git |
| **Packaging** | setuptools, pip |

Dependencies are pinned in `requirements.txt`.

---

## Project Structure

```
chicken_disease_classification/
├── .github/
│   └── workflows/
│       └── main.yaml                   # CI/CD workflow (GitHub Actions)
├── artifacts/                          # Generated outputs (ignored by git)
│   ├── data_ingestion/
│   │   └── Chicken-fecal-images/
│   │       ├── Coccidiosis/
│   │       └── Healthy/
│   ├── prepare_base_model/
│   │   ├── base_model.h5
│   │   └── updated_base_model.h5
│   ├── training/
│   │   └── trained_model.h5
│   └── evaluation/
│       └── scores.json
├── logs/
│   └── running_logs.log                # Central logging output
├── .dvc/                               # DVC metadata (summarized)
│   └── ...                              # Internal DVC files (cache, tmp, config)
├── .dvcignore                          # Patterns for DVC to ignore
├── .gitignore                          # Patterns for git to ignore
├── config/
│   └── config.yaml                     # Artifact paths, dataset source URL
├── src/cnnClassifier/
│   ├── __init__.py                     # Package initialization + logging setup
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # Download & extract dataset
│   │   ├── prepare_base_model.py       # Load VGG16, attach head
│   │   ├── training.py                 # Train with augmentation & validation
│   │   └── evaluation.py               # Evaluate and save metrics
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
├── templates/
│   └── index.html                      # Flask UI (HTML + JavaScript)
├── app.py                              # Flask web app (routes: /, /train, /predict)
├── main.py                             # Pipeline orchestrator (runs all stages)
├── dvc.yaml                            # DVC pipeline definition
├── dvc.lock                            # DVC pipeline lockfile
├── params.yaml                         # Hyperparameters (EPOCHS, BATCH_SIZE, etc)
├── requirements.txt                    # Pinned Python dependencies
├── setup.py                            # Package configuration (pip install -e .)
├── Dockerfile                          # Container image for app server
├── README.md                           # Project documentation
├── cnnClassifier.egg-info/             # Auto-generated package metadata
└── input_image.jpg                     # Default input image placeholder
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
from pathlib import Path
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig, PrepareBaseModelConfig, PrepareCallbacksConfig,
    TrainingConfig, EvaluationConfig
)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root], verbose=False)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config.data_ingestion
        create_directories([c.root_dir], verbose=False)
        return DataIngestionConfig(
            root_dir=Path(c.root_dir),
            source_URL=c.source_URL,
            local_data_file=Path(c.local_data_file),
            unzip_dir=Path(c.unzip_dir)
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        c = self.config.prepare_base_model
        create_directories([c.root_dir], verbose=False)
        return PrepareBaseModelConfig(
            root_dir=Path(c.root_dir),
            base_model_name=Path(c.base_model_name),
            updated_base_model_name=Path(c.updated_base_model_name),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
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
import urllib.request as request
import zipfile
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.source_URL}")
            filename, headers = request.urlretrieve(self.config.source_URL, self.config.local_data_file)
            logger.info(f"{filename} downloaded with info: \n{headers}")
        else:
            logger.info("File already exists")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file to {unzip_path}")
```

**Design Benefits:**
- Single Responsibility: each method has one job
- Reusable: can test `download_file()` and `extract_zip_file()` independently
- Configuration-driven: paths come from config, not hardcoded

---

### 3. Model Preparation (Transfer Learning Setup)

**Location:** `src/cnnClassifier/components/prepare_base_model.py`

```python
import tensorflow as tf
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False
        )
        return self.model

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        code = model(inputs, training=False)
        theta = tf.keras.layers.GlobalAveragePooling2D()(code)
        output = tf.keras.layers.Dense(classes, activation="softmax")(theta)
        full_model = tf.keras.models.Model(inputs, output)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        return full_model
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
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config):
        self.config = config

    def get_updated_data_generator_instance(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode="nearest",
            validation_split=0.2
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = train_datagen.flow_from_directory(
            directory=str(self.config.training_data), subset="training", shuffle=True, class_mode="categorical", **dataflow_kwargs
        )
        valid_gen = valid_datagen.flow_from_directory(
            directory=str(self.config.training_data), subset="validation", shuffle=False, class_mode="categorical", **dataflow_kwargs
        )
        return train_gen, valid_gen

    def train(self, callbacks_list: list):
        model = tf.keras.models.load_model(self.config.updated_base_model_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"]) 
        train_gen, valid_gen = self.get_updated_data_generator_instance()
        model.fit(train_gen, epochs=self.config.params_epochs,
                  steps_per_epoch=len(train_gen), validation_steps=len(valid_gen),
                  validation_data=valid_gen, callbacks=callbacks_list)
        model.save(Path(self.config.root_dir) / self.config.trained_model_name)
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
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
from src.cnnClassifier.pipeline.stage_05_predict import PredictionPipeline
from src.cnnClassifier.utils.common import decodeImage

app = Flask(__name__)
CORS(app)

class clientApp:
    def __init__(self):
        self.filename = "input_image.jpg"
        self.classifier = PredictionPipeline(filename=self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    if request.method == 'POST':
        os.system('python3 main.py')
        return "Training successful!!"
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.get_json(force=True)
    image_b64 = data.get('image') if data else None
    if not image_b64:
        return jsonify({"error": "no image provided"}), 400
    decodeImage(image_b64, client_app.filename)
    result = client_app.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    client_app = clientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)
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

from typing import Any
import shutil
import tensorflow as tf
from box import ConfigBox
from src.cnnClassifier import logger

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logger.info(f"yaml file: {path_to_yaml} loaded successfully")
        return ConfigBox(content)

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        data = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(data)

def save_model(model, path: Path):
    model.save(path)
    logger.info(f"model saved at: {path}")

def load_model(path: Path):
    from tensorflow.keras.models import load_model as keras_load_model
    model = keras_load_model(path)
    logger.info(f"model loaded from: {path}")
    return model

def get_class_names_from_directory(directory: Path) -> list:
    class_names = sorted([item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))])
    logger.info(f"class names found: {class_names}")
    return class_names


def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def decodeImage(image_base64: str, output_path: str):
    if image_base64.startswith('data:'):
        try:
            _, image_base64 = image_base64.split(',', 1)
        except ValueError:
            pass
    image_bytes = base64.b64decode(image_base64)
    with open(output_path, 'wb') as f:
        f.write(image_bytes)
    logger.info(f"Image decoded and saved to: {output_path}")
```

**Benefits:**
- **DRY Principle**: Reuse these functions across all components
- **Centralized Logic**: Change implementation once, affects entire codebase
- **Testability**: Each function is independently testable

---

## Configuration & Hyperparameters

### Main Configuration (`config/config.yaml`)

Defines artifact roots, dataset source, and stage paths:
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: file:///path/to/Chicken-fecal-images.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_name: base_model.h5
  updated_base_model_name: updated_base_model.h5

prepare_callback:
  root_dir: artifacts/prepare_callbacks
  tensorboard_root_log_dir: artifacts/prepare_callbacks/tensorboard_log_dir
  checkpoint_dir: artifacts/prepare_callbacks/checkpoint_dir

training:
  root_dir: artifacts/training
  trained_model_name: trained_model.h5
  training_data: artifacts/data_ingestion/Chicken-fecal-images

evaluation:
  root_dir: artifacts/evaluation
  training_data: artifacts/data_ingestion/Chicken-fecal-images
  mlflow_uri: https://dagshub.com/username/chicken_disease_classification.mlflow
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
[
  { "image": "Healthy" }
]
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
- Classify image (returns disease prediction)
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
dvc add artifacts/data_ingestion/data.zip
git add artifacts/data_ingestion/data.zip.dvc
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

Repository `Dockerfile`:
```dockerfile
FROM python:3.9-slim
RUN apt-get update -y && apt-get install -y awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "app.py"]
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

## CI/CD (GitHub Actions)
- Workflow file: `.github/workflows/main.yaml`; triggers on push to `main` (ignores `README.md`).
- Jobs:
  - Continuous Integration: checkout, lint, and test placeholders.
  - Continuous Delivery: build and push Docker image to AWS ECR.
  - Continuous Deployment: self-hosted runner pulls latest image and restarts container `cnncls` on port `8080`.
- Required secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME`.

### Workflow excerpt
```yaml
jobs:
  build-and-push-ecr-image:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v1
      - run: |
          docker build -t ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest -f Dockerfile .
          docker push ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

  Continuous-Deployment:
    runs-on: self-hosted
    needs: build-and-push-ecr-image
    steps:
      - uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v1
      - run: |
          docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
          docker ps -q --filter "name=cnncls" | grep -q . && docker stop cnncls && docker rm -fv cnncls || true
          docker run -d -p 8080:8080 --name=cnncls \
            -e AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }} \
            -e AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }} \
            -e AWS_REGION=${{ secrets.AWS_REGION }} \
            ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

### Container image
- `Dockerfile` installs `awscli`, copies the repo, installs requirements, and runs `python3 app.py`.
- Verify deployment on the runner: `curl http://localhost:8080/`.

## Logging
- Configured in `src/cnnClassifier/__init__.py:1-18`; writes to `logs/running_logs.log` and stdout.
```python
import logging, os, sys
logging.basicConfig(
  level=logging.INFO,
  format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
  handlers=[
    logging.FileHandler(os.path.join("logs", "running_logs.log")),
    logging.StreamHandler(sys.stdout)
  ]
)
logger = logging.getLogger("cnnClassifier")
```
- Change level with `logging.basicConfig(level=logging.DEBUG)`.
- Runtime logs also appear in `pipeline_output.log` when scripts are invoked via shell.

## API Response Format
- Endpoint: `POST /predict` with body `{ "image": "<base64 or data URI>" }`.
- Example (curl):
```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."}'
```
- Response:
```json
[{ "image": "Healthy" }]
```
- Base64 data URIs are handled by `decodeImage` in `src/cnnClassifier/utils/common.py:58-69` (strips `data:` header and decodes).
- Browser clients are supported via CORS (`Flask-Cors` is enabled in `app.py`).

## DVC Pipeline Stages
- Defined in `dvc.yaml`: `data_ingestion`, `prepare_base_model`, `training`, `evaluation`.
- Run with caching:
```bash
dvc repro --dry-run   # preview
dvc repro             # execute
```
- Each stage executes its corresponding pipeline script via `python -c '...'` as specified in `dvc.yaml`.

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

This project is licensed under the **MIT License**.

## Author & Contact

- **Harsh Goyal** — [@Harshgoyal2004](https://github.com/Harshgoyal2004)
- Issues & Contributions welcome on [GitHub](https://github.com/Harshgoyal2004/chicken_disease_classification)

---

**Last Updated**: November 18, 2025  
**Model Accuracy**: 88.72%  
**Status**: Production-Ready ✓
