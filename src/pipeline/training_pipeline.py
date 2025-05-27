from hotelres.data_ingestion import DataIngestion
from hotelres.data_preprocessing import DataProcessor
from hotelres.model_training import ModelTraining
from utils.commonfunctions import read_yaml
from config.paths_config import *



if __name__ == "__main__":

    ## DATA Ingetion
    dataingestion = DataIngestion(read_yaml(CONFIG_PATH))
    dataingestion.run()

    ## DATA PREPROCESSINGS
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()

    ## Model TRAININGS 
    trainer =ModelTraining(PROCESSED_TRAIN,PROCESSED_TEST,MODEL_OUTPUT_PATH)
    trainer.run()