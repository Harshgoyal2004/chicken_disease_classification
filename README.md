# Chicken Disease Classification (Comprehensive)

A full, self-contained developer and interviewer-friendly reference for the Chicken Disease Classification pipeline (Coccidiosis vs Healthy). This README explains the problem, the design, the implementation, how to run everything, and what to say in an interview.

TABLE OF CONTENTS
-----------------
1. Project summary (elevator pitch)
2. Why it matters
3. Repo layout and important files
4. Dataset and expected layout
5. Architecture and model design
6. Pipeline stages (detailed)
7. Configuration and hyperparameters
8. How to run (commands)
9. DVC and artifact tracking
10. Troubleshooting and common fixes
11. Evaluation and interpreting results
12. How to present this project in an interview (talking points)
13. Extensions and next steps
14. Reproducibility checklist
15. References, license, contact

1) Project summary (elevator pitch)
----------------------------------
This project implements an automated image classifier to detect Coccidiosis (a parasitic infection) from chicken fecal images. It uses transfer learning (VGG16 backbone) and an engineered pipeline with modular stages (ingest, prepare base model, training, evaluation). The repo also includes a lightweight Flask front-end for inference and DVC metadata to enable reproducible experiments.

2) Why it matters
-----------------
- Automation reduces manual workload for farmers and veterinarians and enables rapid screening.
- A small model and simple UI can be deployed on edge devices or low-cost servers for real-time assistance.

3) Repo layout and important files
---------------------------------
- `main.py` — orchestrates the full pipeline.
- `app.py` + `templates/index.html` — minimal Flask app for uploading images and getting predictions.
- `config/config.yaml` — artifact paths and dataset source.
- `params.yaml` — hyperparameters used by the pipeline.
- `dvc.yaml`, `.dvc/` — DVC pipeline and metadata (for data/model artifact tracking).
- `src/cnnClassifier/`:
  - `components/` — core pipeline modules (`data_ingestion.py`, `prepare_base_model.py`, `training.py`, `evaluation.py`).
  - `pipeline/` — small scripts that run each stage.
  - `config/` — configuration manager module.
  - `utils/common.py` — helpers used across the project (YAML parsing, saving/loading models, `decodeImage`).

4) Dataset and expected layout
------------------------------
The pipeline expects a zip with the following layout when extracted:

```
artifacts/data_ingestion/Chicken-fecal-images/
  ├─ Coccidiosis/
  │   ├ image1.jpg
  │   └ ...
  └─ Healthy/
      ├ imageX.jpg
      └ ...
```

Place `Chicken-fecal-images.zip` in the repo root or change `config.data_ingestion.source_URL` to a local file URI.

5) Architecture and model design
--------------------------------
- Backbone: VGG16 (ImageNet weights, `include_top=False`).
- Head: GlobalAveragePooling2D -> Dense(256, relu) -> Dropout -> Dense(num_classes, softmax).
- Training: Adam optimizer, CategoricalCrossentropy loss, metrics=['accuracy'].
- Data augmentation: rotation, width/height shift, shear, zoom, horizontal flip.

Why VGG16?
- VGG16 is simple and well-known; good baseline for transfer learning. It's easy to explain in interviews and effective for small-scale problems. For production you can upgrade to EfficientNet/ResNet.

6) Pipeline stages (detailed)
----------------------------
Stage 1 — Data Ingestion
- Downloads or reads local zip (`source_URL`) and extracts to `artifacts/data_ingestion`.

Stage 2 — Prepare Base Model
- Loads VGG16 without top layers; attaches classification head; compiles a model with initial frozen base layers and saves `base_model.h5` and `updated_base_model.h5`.

Stage 3 — Training
- Uses `ImageDataGenerator(..., validation_split=0.2)` and `flow_from_directory` with `subset='training'` and `subset='validation'`.
- Reloads `updated_base_model.h5`, recompiles (important to avoid Keras eager/compile mismatches), and runs `model.fit`.

Stage 4 — Evaluation
- Builds a validation generator (same `validation_split`) and evaluates the trained model. Results are saved to `artifacts/evaluation/scores.json`.

Stage 5 — Predict
- A thin wrapper used by the Flask app — decodes incoming base64 image, loads `artifacts/training/trained_model.h5` and returns predicted label.

7) Configuration and hyperparameters
------------------------------------
- `config/config.yaml` defines artifact roots and dataset location.
- `params.yaml` controls: `AUGMENTATION`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `CLASSES`, `CLASS_NAMES`.

8) How to run (commands)
-------------------------
Clone & setup
```bash
git clone https://github.com/Harshgoyal2004/chicken_disease_classification.git
cd chicken_disease_classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Quick smoke-test (1-epoch training)
```bash
# ensure params.yaml sets EPOCHS: 1
python -c "from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline; TrainingPipeline().main()"
```

Run full pipeline
```bash
python main.py
```

Start Flask UI
```bash
python app.py
# Visit http://127.0.0.1:8080
```

9) DVC and artifact tracking
----------------------------
- `dvc.yaml` describes the pipeline stages. Use `dvc repro` to run stages under DVC control.
- Track large files:

```bash
dvc add artifacts/data_ingestion/data.zip
git add artifacts/data_ingestion/data.zip.dvc
git commit -m "Add dataset to DVC"
dvc remote add -d myremote s3://<bucket>/path
dvc push
```

10) Troubleshooting and common fixes
-----------------------------------
- "The PyDataset has length 0": Ensure `ImageDataGenerator(..., validation_split=0.2)` is set and `flow_from_directory(..., subset='validation')` uses the same `validation_split`.
- Keras compile/eager errors: After loading a saved model, recompile it before training or evaluation.
- Model save format: Keras may warn about HDF5 being legacy — use `.keras` for native format if desired.

11) Evaluation and interpreting results
--------------------------------------
- Primary metrics saved in `artifacts/evaluation/scores.json`.
- For thorough evaluation compute: accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC if using probabilities.

12) How to present this project in an interview (talking points)
------------------------------------------------------------
Use the following structure when speaking:

- Problem: "What and why" (one-liner)
- Data: "Where it comes from, size, structure, challenges (class balance, image quality)"
- Approach: "Transfer learning with VGG16; augmentations; validation split"
- Engineering: "Config-driven pipeline, DVC for large artifacts, Flask UI for demo"
- Results: "Give evaluated metrics and caveats" (refer to `artifacts/evaluation/scores.json`)
- Next steps: "Collect more data, fine-tune backbone, add explainability, dockerize"

Example short pitch you can memorize
------------------------------------
"I built an end-to-end transfer-learning pipeline that classifies chicken fecal images for Coccidiosis. The system uses VGG16 as a backbone, a configurable training pipeline with augmentation and validation splitting, and DVC for artifact tracking. I also added a minimal web UI for quick demos and kept the code modular so each stage can be run independently or under DVC." 

13) Extensions and next steps
---------------------------
- Replace backbone with EfficientNet or ResNet and compare performance.
- Implement cross-validation and hyperparameter sweeps (Optuna or Keras Tuner).
- Add Grad-CAM explanations and a simple dashboard to visualize model decisions.
- Containerize with Docker and deploy behind a WSGI server.

14) Reproducibility checklist
---------------------------
1. `venv` + pinned `requirements.txt`
2. `config/config.yaml` and `params.yaml` documented and checked into the repo
3. Use DVC for raw data and large artifacts
4. Save model weights and `scores.json` artifacts under `artifacts/`

15) References, license, contact
--------------------------------
- VGG16 paper: https://arxiv.org/abs/1409.1556
- TF transfer learning guide: https://www.tensorflow.org/tutorials/images/transfer_learning

License: MIT

Author & Contact
- Harsh Goyal — https://github.com/Harshgoyal2004

---

EXTENDED SECTION: Complete Project Overview
============================================

16) Complete project file structure
----------------------------------
```
chicken_disease_classification/
├── artifacts/                          # Generated outputs (not committed)
│   ├── data_ingestion/
│   │   └── Chicken-fecal-images/
│   │       ├── Coccidiosis/            # Class folder with images
│   │       └── Healthy/                # Class folder with images
│   ├── prepare_base_model/
│   │   ├── base_model.h5               # VGG16 without top (download + freeze)
│   │   └── updated_base_model.h5       # VGG16 + custom head (compiled, trainable)
│   ├── training/
│   │   └── trained_model.h5            # Final trained model after 1 epoch
│   └── evaluation/
│       └── scores.json                 # Evaluation metrics {loss, accuracy}
│
├── .dvc/                               # DVC metadata (tracked in git after `dvc init`)
│   ├── config
│   └── .gitignore
│
├── .dvcignore                          # DVC ignore patterns
├── .gitignore                          # Git ignore (excludes venv/, artifacts/, etc)
│
├── config/
│   └── config.yaml                     # Artifact paths, dataset source URL
│
├── logs/                               # Optional: log output directory
│
├── src/
│   └── cnnClassifier/
│       ├── __init__.py                 # Package init + logger setup
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py       # Download & extract dataset
│       │   ├── prepare_base_model.py   # Load VGG16, build head, save model
│       │   ├── training.py             # Data gen, load model, fit, save
│       │   └── evaluation.py           # Build val gen, evaluate, save scores
│       ├── config/
│       │   ├── __init__.py
│       │   └── configuration.py        # ConfigurationManager: loads YAML & returns dataclass configs
│       ├── constants/
│       │   └── __init__.py             # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│       ├── entity/
│       │   ├── __init__.py
│       │   └── config_entity.py        # Dataclasses: DataIngestionConfig, TrainingConfig, etc.
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── stage_01_data_ingestion.py      # Run: DataIngestionPipeline().main()
│       │   ├── stage_02_prepare_base_model.py  # Run: PrepareBaseModelPipeline().main()
│       │   ├── stage_03_training.py            # Run: TrainingPipeline().main()
│       │   ├── stage_04_evaluation.py          # Run: EvaluationPipeline().main()
│       │   └── stage_05_predict.py             # Used by Flask app for inference
│       └── utils/
│           ├── __init__.py
│           └── common.py               # Helper functions: read_yaml, save_model, load_model, decodeImage
│
├── templates/
│   └── index.html                      # Flask UI: upload image, preview, predict
│
├── main.py                             # Orchestrator: runs all stages in sequence
├── app.py                              # Flask web app with /train and /predict routes
├── setup.py                            # Package setup (installable with pip)
├── requirements.txt                    # Python package dependencies
├── config.yaml                         # Artifact roots (can be symlinked or copied)
├── params.yaml                         # Hyperparameters (EPOCHS, BATCH_SIZE, etc)
├── dvc.yaml                            # DVC pipeline definition (stages: data_ingestion, prepare_base_model, training, evaluation)
├── README.md                           # This comprehensive guide
└── cnnClassifier.egg-info/             # Generated by pip install -e . (metadata)

Key files to understand:
- src/cnnClassifier/__init__.py — Python logger setup
- src/cnnClassifier/utils/common.py — Reusable utilities
- src/cnnClassifier/config/configuration.py — Single source of truth for all configs
- main.py — High-level orchestrator
- app.py — Flask app entry point
```

17) Technology stack
-------------------
Language & Core:
- Python 3.9+
- pip + setuptools (packaging)

Deep Learning & ML:
- TensorFlow >= 2.10.0 (Keras API)
- NumPy — numerical arrays
- Scikit-learn — ML utilities (metrics, preprocessing)
- OpenCV (cv2) — image I/O and processing
- Pillow — image handling

Configuration & Utilities:
- PyYAML — YAML parsing
- Python-Box — nested dict access via dot notation

Web & Serving:
- Flask — lightweight web framework
- Werkzeug — WSGI utilities (integrated with Flask)

Reproducibility & Pipelines:
- DVC (Data Version Control) — artifact tracking, pipeline orchestration
- Git — version control

Development:
- pip-tools (optional) — pinned dependencies
- pytest (optional) — testing

All dependencies are pinned in `requirements.txt` for reproducibility.

18) Important code samples
--------------------------

A) Configuration Management (src/cnnClassifier/config/configuration.py)
-----------

```python
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
)

class ConfigurationManager:
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

Why this pattern?
- Single source of truth for all paths and hyperparameters.
- Dataclasses ensure type safety and immutability.
- Easy to swap configs for different environments (dev, prod).

B) Data Ingestion Component (src/cnnClassifier/components/data_ingestion.py)
-----------

```python
import os
import zipfile
import urllib.request
from pathlib import Path
from src.cnnClassifier.utils.common import get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = urllib.request.urlretrieve(
                self.config.source_URL,
                self.config.local_data_file
            )
            print(f"Downloaded {filename} ({get_size(filename)})")
        else:
            print(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_path
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as z:
            z.extractall(unzip_path)
        print(f"Extracted to {unzip_path}")
```

Why this design?
- Separation of concerns: one method for download, one for extraction.
- Re-usable and testable in isolation.
- Config drives behavior (no hardcoded paths).

C) Model Preparation (src/cnnClassifier/components/prepare_base_model.py)
-----------

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from src.cnnClassifier.utils.common import save_model, load_model

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = VGG16(
            input_shape=self.config.params.IMAGE_SIZE,
            weights='imagenet',
            include_top=False
        )
        save_model(self.model, self.config.base_model_path)
        return self.model

    def _prepare_full_model(self, learning_rate):
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
        self.full_model = self._prepare_full_model(
            learning_rate=self.config.params.LEARNING_RATE
        )
        save_model(self.full_model, self.config.updated_base_model_path)
```

Why VGG16?
- Simple, interpretable architecture (16 conv layers).
- Pre-trained ImageNet weights transfer well to small datasets.
- Good baseline for interview explanations.

D) Training Pipeline (src/cnnClassifier/components/training.py)
-----------

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.cnnClassifier.utils.common import load_model

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_updated_data_generator_instance(self, data_path):
        datagenerator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Critical for split
        )
        return datagenerator.flow_from_directory(
            directory=data_path,
            target_size=self.config.params.IMAGE_SIZE[:2],
            batch_size=self.config.params.BATCH_SIZE,
            class_mode='categorical',  # Important for multi-class
            validation_split=0.2  # Must match above
        )

    def train(self):
        self.model = load_model(self.config.updated_base_model_path)
        # Recompile after load to avoid eager execution issues
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

Key gotchas handled:
- `validation_split=0.2` splits 80/20 inside the generator.
- `class_mode='categorical'` ensures one-hot encoded labels for multi-class.
- Recompile after load to avoid Keras cached metadata issues.

E) Flask Web App (app.py)
-----------

```python
from flask import Flask, render_template, request, jsonify
from src.cnnClassifier.pipeline.stage_05_predict import PredictionPipeline
import os
from src.cnnClassifier.utils.common import decodeImage

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train_route():
    os.system("python3 main.py")
    return jsonify({"status": "Training started"})

@app.route("/predict", methods=["POST"])
def predict_route():
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode base64 to file
    decodeImage(image_data, "input_image.jpg")
    
    # Predict
    pipeline = PredictionPipeline("input_image.jpg")
    result = pipeline.predict()
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
```

Why Flask?
- Lightweight, no magic, easy to understand and extend.
- Perfect for prototyping and demos.
- For production, wrap with Gunicorn or uWSGI.

F) Helper Utilities (src/cnnClassifier/utils/common.py)
-----------

```python
import yaml
import json
import os
import base64
from pathlib import Path

def read_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    model.save(path)
    print(f"Model saved at {path}")

def load_model(path):
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(path)

def decodeImage(img_string, filename):
    """Decode base64 image string and save to disk."""
    img_data = base64.b64decode(img_string)
    with open(filename, 'wb') as f:
        f.write(img_data)
    print(f"Image decoded and saved to {filename}")

def create_directories(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
```

Why separate utilities?
- Single responsibility: each function does one thing.
- Reusable across all components.
- Easy to unit test.

19) How each component works together
------------------------------------

Step-by-step flow:
1. User runs `python main.py`.
2. `main.py` calls each stage in order.
3. Stage 1 (Data Ingestion):
   - ConfigurationManager reads `config.yaml` → DataIngestionConfig.
   - DataIngestion.download_file() gets the dataset zip.
   - DataIngestion.extract_zip_file() extracts to `artifacts/data_ingestion/Chicken-fecal-images/`.
4. Stage 2 (Prepare Base Model):
   - ConfigurationManager reads config → PrepareBaseModelConfig.
   - PrepareBaseModel.get_base_model() loads VGG16 (no top) and saves to `base_model.h5`.
   - PrepareBaseModel.update_base_model() attaches head and saves to `updated_base_model.h5`.
5. Stage 3 (Training):
   - ConfigurationManager reads config + params → TrainingConfig.
   - Training loads `updated_base_model.h5` and recompiles.
   - Training.get_updated_data_generator_instance() creates 80/20 split generators.
   - Training.train() runs `model.fit()` for N epochs (default 25, set to 1 for testing).
   - Saves trained model to `trained_model.h5`.
6. Stage 4 (Evaluation):
   - Evaluation creates same generator (validation_split=0.2).
   - Loads `trained_model.h5` and calls `model.evaluate()` on validation set.
   - Saves loss + accuracy to `artifacts/evaluation/scores.json`.
7. Stage 5 (Predict, used by Flask):
   - PredictionPipeline loads `trained_model.h5`.
   - Loads an image, normalizes, and calls `model.predict()`.
   - Returns class name + confidence.

Why this architecture?
- **Modularity**: Each component is independent, testable, reusable.
- **Configuration-driven**: Change `params.yaml` → all stages adapt. No code edits needed.
- **Clear separation of concerns**: Components handle one task; pipelines orchestrate.
- **Reproducibility**: All paths and hyperparameters are documented in YAML files.

20) Key design patterns used
----------------------------
1. **Configuration Manager Pattern**
   - Single source of truth for all settings.
   - Dataclasses ensure type safety.
   - Example: `ConfigurationManager.get_training_config()` returns immutable TrainingConfig.

2. **Pipeline Pattern**
   - Each stage (data_ingestion, prepare_base_model, training, evaluation) is a standalone pipeline.
   - Pipelines can be run individually or orchestrated by `main.py` or DVC.

3. **Component Pattern**
   - Core logic lives in components (DataIngestion, Training, etc.).
   - Pipelines wrap components and handle setup/teardown.

4. **Utility/Helper Functions**
   - Common operations (YAML I/O, model save/load) are extracted to `utils/common.py`.
   - Reduces duplication and centralizes logic.

5. **Entity/DTO Pattern**
   - Dataclasses (DataIngestionConfig, TrainingConfig, etc.) act as transfer objects.
   - Type hints and validation improve code clarity and IDE support.

6. **DVC Pipeline Pattern**
   - `dvc.yaml` describes dependencies between stages (e.g., training depends on prepare_base_model).
   - `dvc repro` re-runs stages if inputs change (outputs are cached).

21) Typical workflow example (for interview)
--------------------------------------------

Scenario: You want to experiment with a smaller batch size.

1. Open `params.yaml`:
   ```yaml
   BATCH_SIZE: 16  # Changed from 32
   ```
2. Run training:
   ```bash
   python -c "from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline; TrainingPipeline().main()"
   ```
3. ConfigurationManager automatically reads the new BATCH_SIZE.
4. Training uses batch size 16, runs, and saves results.
5. Evaluation picks up the new model and reports scores.

No code changes needed — config drives behavior.

22) Metrics and results interpretation
------------------------------------
After evaluation, check `artifacts/evaluation/scores.json`:
```json
{
  "loss": 0.2792516350746155,
  "accuracy": 0.8871794939041138
}
```

Interpretation:
- Loss (0.279): Lower is better. Measure of prediction error. Still reasonable for a simple model.
- Accuracy (88.72%): 88.72% of validation images are correctly classified. Good baseline.

If accuracy is low:
- Increase EPOCHS in `params.yaml`.
- Collect more data (class imbalance?).
- Fine-tune the backbone (set base layers to trainable).
- Try a deeper model (ResNet, EfficientNet).

23) Deployment considerations
----------------------------
For production:
1. **Containerization**: Create a Dockerfile to bundle the app + dependencies.
   ```dockerfile
   FROM python:3.9
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt && pip install -e .
   CMD ["python", "app.py"]
   ```
2. **WSGI Server**: Use Gunicorn or uWSGI instead of Flask's dev server.
   ```bash
   gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
   ```
3. **Model Versioning**: Store models in a model registry (MLflow, BentoML).
4. **Monitoring**: Log predictions, latencies, and errors.
5. **Scalability**: Use DVC to manage large artifacts; sync models from cloud storage.

If you want I will also create a short `slides.md` summarizing the project in 6–8 slides (bullet points) that you can use to practice interview responses.
