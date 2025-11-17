from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    training = TrainingPipeline()
    training.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    evaluation = EvaluationPipeline()
    evaluation.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

