from pathlib import Path
from src.cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root], verbose=False)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir], verbose=False)
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir], verbose=False)

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_name=Path(config.base_model_name),
            updated_base_model_name=Path(config.updated_base_model_name),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
        )
        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callback
        model_ckpt_dir = Path(config.checkpoint_dir)
        tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir)
        
        create_directories([
            Path(tensorboard_root_log_dir),
            model_ckpt_dir
        ], verbose=False)

        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=tensorboard_root_log_dir,
            checkpoint_model_filepath=model_ckpt_dir / "model.h5"
        )
        return prepare_callbacks_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        create_directories([training.root_dir], verbose=False)

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_name=Path(training.trained_model_name),
            updated_base_model_path=Path(prepare_base_model.root_dir) / prepare_base_model.updated_base_model_name,
            training_data=Path(training.training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation
        create_directories([eval_config.root_dir], verbose=False)

        evaluation_config = EvaluationConfig(
            path_of_model=Path(self.config.training.root_dir) / self.config.training.trained_model_name,
            training_data=Path(eval_config.training_data),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            mlflow_uri=eval_config.mlflow_uri
        )
        return evaluation_config
