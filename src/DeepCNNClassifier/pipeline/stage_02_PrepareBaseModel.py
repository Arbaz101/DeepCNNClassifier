from DeepCNNClassifier.config import ConfigurationManager
from DeepCNNClassifier.components import PrepareBaseModel
from DeepCNNClassifier import logger

STAGE_NAME = "Prepare Base Model"

def main():
    try:
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
    except Exception as e:
        raise e

if __name__ == '__main__':
    try:
        logger.info("\n\n")
        logger.info("\n\n")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e