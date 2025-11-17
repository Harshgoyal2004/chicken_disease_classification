import os
import urllib.request as request
import zipfile
from src.cnnClassifier import logger
from src.cnnClassifier.utils.common import get_class_names_from_directory
from src.cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.source_URL}")
            filename, headers = request.urlretrieve(
                self.config.source_URL,
                self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file to {unzip_path}")
