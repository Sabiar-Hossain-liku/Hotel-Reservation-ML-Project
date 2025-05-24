import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from hotelres.logger import get_logger  
from hotelres.custom_exception import CustomException
from hotelres.config.paths_config import *
from hotelres.utils.commonfunctions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):

        # this belog to config.yaml
        self.config = config['data_ingestion'] 
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_test_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.file_name}")

    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            blub = bucket.blob(self.file_name)
            blub.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV FILE is downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while downloading CSV file")
            raise CustomException("Filld to download the file",e)
        
    

    def split_data(self):

        try:
            logger.info("Starting the splitting the process")
            
            data =  pd.read_csv(RAW_FILE_PATH)
            train_data, test_data =train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error whil splitting the data")
            raise CustomException("Faild to split the data",e)
        

    def run(self):

        try:

            logger.info("Starting the data ingestion  process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        
        finally:
            logger.info("data Ingestion Process Completed")


if __name__ =="__main__":

    dataingestion = DataIngestion(read_yaml(CONFIG_PATH))
    dataingestion.run()

