import os
from pathlib import Path
import tensorflow as tf
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from src.cnnClassifier.utils.common import save_model

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False
        )
        self.save_model(path=self.config.root_dir, model_name=self.config.base_model_name, model=self.model)
        logger.info(f"Base model {self.config.base_model_name} saved")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

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

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.root_dir, model_name=self.config.updated_base_model_name, model=self.full_model)
        logger.info(f"Updated model {self.config.updated_base_model_name} saved")

    @staticmethod
    def save_model(path, model_name, model=None):
        model_path = os.path.join(path, model_name)
        model.save(model_path)
