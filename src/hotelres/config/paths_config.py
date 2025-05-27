import os


####################### DATA INGETION ###############################


RAW_DIR = "artifacts/raw"
RAW_FILE_PATH= os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH =os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "src/hotelres/config/config.yaml"

##################### DATA PROCESSING ###################################


PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST = os.path.join(PROCESSED_DIR, "processed_test.csv")


####################### MODEL TRAINING ################################

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"