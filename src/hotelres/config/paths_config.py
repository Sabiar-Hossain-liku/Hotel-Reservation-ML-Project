import os


####################### DATA INGETION ###############################


RAW_DIR = "artifacts/raw"
RAW_FILE_PATH= os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH =os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "src/hotelres/config/config.yaml"