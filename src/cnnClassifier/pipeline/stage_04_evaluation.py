from src.cnnClassifier import logger
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.evaluation import Evaluation

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.save_score()
