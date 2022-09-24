from DeepCNNClassifier.constants import *
from DeepCNNClassifier.utils import read_yaml, create_directories
from DeepCNNClassifier.entity import DataIngestionConfig
from DeepCNNClassifier import logger

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        logger.info(f"Data Ingestion Config is completed: {data_ingestion_config}")
        return data_ingestion_config