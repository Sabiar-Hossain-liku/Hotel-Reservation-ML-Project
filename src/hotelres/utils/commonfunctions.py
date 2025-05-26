import os
import pandas
from hotelres.logger import get_logger
from hotelres.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the path")
        
        # Opening the yaml File to Read
        with open(file_path, "r") as yaml_file:
            
            config = yaml.safe_load(yaml_file)
            logger.info("Reading yaml file")
            
            return config
        
    except Exception as e:
        logger.error("Error in the file reading")
        raise CustomException("Failed to read Yaml file ", e)
    

def load_data(path):
    try:
        logger.info("loading Data...")
        data =pd.read_csv(path)
        logger.info("data loaded")
        return data
    except Exception as e:
        logger.error("Error loading the csv file")
        raise CustomException("Failled loading the Csv",e)
    

