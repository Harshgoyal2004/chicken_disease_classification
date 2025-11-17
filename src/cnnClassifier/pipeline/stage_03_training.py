from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.training import Training

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train(callbacks_list=[])
