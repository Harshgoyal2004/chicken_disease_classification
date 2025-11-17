import os
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier import logger
from src.cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_updated_data_generator_instance(self, train_data_gen_config):
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
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="training",
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )
        return train_generator, valid_generator

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info(f"model saved at {path}")

    def train(self, callbacks_list: list):
        self.steps_per_epoch = self.config.params_batch_size
        self.validation_steps = self.config.params_batch_size

        # Load model fresh and recompile
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        train_generator, valid_generator = self.get_updated_data_generator_instance(
            train_data_gen_config=None
        )

        self.steps_per_epoch = len(train_generator)
        self.validation_steps = len(valid_generator)

        self.model.fit(
            train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=valid_generator,
            callbacks=callbacks_list
        )

        self.save_model(
            path=self.config.root_dir / self.config.trained_model_name,
            model=self.model
        )
