# Chicken Disease Classification

A deep learning project to classify chicken fecal images to detect disease (Coccidiosis) using Transfer Learning with VGG16.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to automatically classify chicken fecal images into two categories:
- **Healthy**: Normal chicken feces
- **Coccidiosis**: Disease-infected chicken feces

The model achieves **88.72% accuracy** on the validation dataset using VGG16 transfer learning.

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 68.20% |
| Validation Accuracy | 88.75% |
| Final Evaluation Accuracy | 88.72% |
| Validation Loss | 0.279 |

## 🏗️ Project Architecture

```
chicken_disease_classification/
├── src/cnnClassifier/
│   ├── components/           # Core components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── training.py
# Chicken Disease Classification

A deep learning pipeline to classify chicken fecal images (Coccidiosis vs Healthy) using TensorFlow/Keras and VGG16 transfer learning.

**This repository contains the pipeline code only — the dataset is not included.**

**Quick summary**
- Stages: Data ingestion → Prepare base model → Training → Evaluation
- Config-driven via `config/config.yaml` and `params.yaml`
- Artifacts saved under `artifacts/`

---

## Quick Start

Prerequisites
- Python 3.8+ (3.9 recommended)
- `git`, `pip`

Setup
```bash
git clone https://github.com/Harshgoyal2004/chicken_disease_classification.git
cd chicken_disease_classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Dataset
- Place `Chicken-fecal-images.zip` in the project root, or update `data_ingestion.source_URL` in `config/config.yaml` to point to your zip (e.g. `file:///full/path/to/Chicken-fecal-images.zip`).
- The data ingestion stage will extract images into `artifacts/data_ingestion/Chicken-fecal-images/`.

Important: the repository `.gitignore` excludes `venv/` and model/artifact files — do not commit the dataset or artifacts.

---

## Running the pipeline

Run full pipeline (all stages):
```bash
python main.py
```

Run individual stages from Python (useful for development):
```python
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

DataIngestionPipeline().main()
PrepareBaseModelPipeline().main()
TrainingPipeline().main()
EvaluationPipeline().main()
```

Run a quick test (1-epoch training)
- Edit `params.yaml` and set `EPOCHS: 1`, or it may already be set for quick runs.
- Then run only the training stage to validate the end-to-end flow:
```bash
python -c "from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline; TrainingPipeline().main()"
```

---

## Configuration

`config/config.yaml` controls artifact paths and the dataset source. Update `data_ingestion.source_URL` if your dataset is stored elsewhere.

`params.yaml` contains hyperparameters. Example (editable):
```yaml
AUGMENTATION: true
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 32
# Set EPOCHS to 1 for a quick smoke-test
EPOCHS: 1
LEARNING_RATE: 0.001
CLASSES: 2
CLASS_NAMES: ['Coccidiosis', 'Healthy']
```

---

## Artifacts produced
- `artifacts/prepare_base_model/` — `base_model.h5`, `updated_base_model.h5`
- `artifacts/training/trained_model.h5` — final trained model
- `artifacts/evaluation/scores.json` — evaluation metrics (loss, accuracy)

Check `artifacts/evaluation/scores.json` after evaluation to see the actual metrics for your run.

---

## Notes & Troubleshooting

- If `flow_from_directory(..., subset='validation')` returns 0 images, ensure the `ImageDataGenerator` in training and evaluation uses the same `validation_split` (the code config includes `validation_split=0.2`).
- If you see `The PyDataset has length 0` during evaluation, run the training stage (which creates the validation split) or verify the `training_data` path in `config/config.yaml` points to the extracted class folders.
- For reproducible experiments, pin package versions in `requirements.txt` and use a consistent `params.yaml`.

---

## Development & Contribution

- The package is installable in editable mode (`pip install -e .`) to allow live edits while developing.
- The `src/` directory contains the implementation. Please open issues or PRs for improvements.

---

## License

This project is MIT licensed — see `LICENSE`.

## Author
- Harsh Goyal — https://github.com/Harshgoyal2004

---

**Last Updated**: 2025-11-18
