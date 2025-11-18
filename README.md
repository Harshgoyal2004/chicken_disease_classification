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

If you want I will also create a short `slides.md` summarizing the project in 6–8 slides (bullet points) that you can use to practice interview responses.
