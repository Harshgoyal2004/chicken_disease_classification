import os
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.utils.common import save_json, load_model, get_class_names_from_directory
from src.cnnClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def get_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs,
            validation_split=0.2
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path):
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.get_valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }
        save_json(path=Path("artifacts/evaluation/scores.json"), data=scores)

    def log_into_mlflow(self):
        pass


