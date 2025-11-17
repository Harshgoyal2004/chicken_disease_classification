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
│   │   └── evaluation.py
│   ├── pipeline/             # Pipeline stages
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   └── stage_04_evaluation.py
│   ├── config/
│   │   └── configuration.py  # Configuration management
│   ├── entity/
│   │   └── config_entity.py  # Config dataclasses
│   ├── utils/
│   │   └── common.py         # Utility functions
│   └── constants/
│       └── __init__.py       # Constants
├── config/
│   └── config.yaml           # YAML configuration
├── artifacts/                # Model artifacts & data
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   ├── training/
│   └── evaluation/
├── main.py                   # Entry point
├── params.yaml               # Model hyperparameters
└── requirements.txt          # Dependencies
```

## 🔧 Technical Stack

- **Framework**: TensorFlow/Keras
- **Transfer Learning**: VGG16 (pre-trained on ImageNet)
- **Data Augmentation**: ImageDataGenerator
- **Language**: Python 3.9
- **Configuration**: YAML-based

## 📋 Pipeline Stages

### Stage 1: Data Ingestion
- Downloads and extracts training dataset
- Organizes images by disease class

### Stage 2: Prepare Base Model
- Loads pre-trained VGG16 model
- Freezes base layers for transfer learning
- Adds custom classification head for 2-class output

### Stage 3: Training
- Applies data augmentation (rotation, zoom, shift)
- Trains the model with Adam optimizer
- Uses early stopping to prevent overfitting
- Saves best model as `trained_model.h5`

### Stage 4: Evaluation
- Evaluates model on validation dataset
- Computes loss and accuracy metrics
- Saves evaluation scores to JSON

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harshgoyal2004/chicken_disease_classification.git
   cd chicken_disease_classification
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in editable mode**
   ```bash
   pip install -e .
   ```

## 🚀 Usage

### Run Full Pipeline
Execute all 4 stages sequentially:
```bash
python main.py
```

### Run Individual Stages
```python
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

# Stage 1: Data Ingestion
data_ingestion = DataIngestionPipeline()
data_ingestion.main()

# Stage 2: Prepare Base Model
prepare_base_model = PrepareBaseModelPipeline()
prepare_base_model.main()

# Stage 3: Training
training = TrainingPipeline()
training.main()

# Stage 4: Evaluation
evaluation = EvaluationPipeline()
evaluation.main()
```

### Use Trained Model for Inference

```python
from src.cnnClassifier.utils.common import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('artifacts/training/trained_model.h5')

# Prepare image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)
class_names = ['Coccidiosis', 'Healthy']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
```

## 📁 Configuration Files

### `config/config.yaml`
Defines artifact paths and data source for all stages

### `params.yaml`
Model hyperparameters:
```yaml
AUGMENTATION: true
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 32
EPOCHS: 25
LEARNING_RATE: 0.001
CLASSES: 2
CLASS_NAMES: ['Coccidiosis', 'Healthy']
```

## 📈 Model Details

### VGG16 Transfer Learning
- **Pre-trained weights**: ImageNet
- **Base layers**: Frozen (no fine-tuning)
- **Custom head**: Dense layers for 2-class classification
- **Output**: Softmax activation for probability distribution

### Data Augmentation
- Rotation range: 20°
- Width/Height shift: 0.2
- Shear range: 0.2
- Zoom range: 0.2
- Horizontal flip: Yes

## 🔍 Evaluation Metrics

Results are saved in `artifacts/evaluation/scores.json`:
```json
{
  "loss": 0.2792516350746155,
  "accuracy": 0.8871794939041138
}
```

## 📦 Dependencies

Key packages:
- `tensorflow>=2.10.0` - Deep learning framework
- `numpy` - Numerical operations
- `pandas` - Data handling
- `opencv-python` - Image processing
- `scikit-learn` - ML utilities
- `pyyaml` - YAML parsing
- `python-box` - Config management

See `requirements.txt` for full list.

## 🐛 Troubleshooting

### Issue: Module not found error
**Solution**: Ensure the package is installed in editable mode:
```bash
pip install -e .
```

### Issue: GPU not detected
**Solution**: For CPU-only training, ensure `tensorflow-cpu` is used or manually set:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue: Out of memory
**Solution**: Reduce batch size in `params.yaml`:
```yaml
BATCH_SIZE: 16  # Reduced from 32
```

## 📝 Logging

The project uses Python's logging module. Check logs in:
- Console output during execution
- Log files in `logs/` directory (if configured)

## 🔗 Related Resources

- [VGG16 Architecture](https://arxiv.org/abs/1409.1556)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

- **Harsh Goyal** - [@Harshgoyal2004](https://github.com/Harshgoyal2004)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Last Updated**: 2024
**Model Accuracy**: 88.72%
