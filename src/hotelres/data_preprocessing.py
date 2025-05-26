import os
import pandas as pd
import numpy as np

from hotelres.logger import get_logger
from hotelres.custom_exception import CustomException
from hotelres.config.paths_config import *
from hotelres.utils.commonfunctions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

#######################################################################

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,Processed_dir,config_path):

        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = Processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self,df):

        try:
            logger.info("Starting our Data Processing step")

            logger.info("Dropping Columns")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_col = self.config["data_processing"]['categorical_columns']
            num_col = self.config['data_processing']['numerical_columns']

            logger.info("Applying label encoding")
            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_col:
                df[col] = label_encoder.fit_transform(df[col])
                
                mappings[col] = dict(zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)))

                logger.info("Label Mapping are: ")
                for col,mappings in mappings.items():
                    logger.info(f"{col} : {mappings}")

            
            logger.info("Skewness Handling")

            skewness_threshold =self.config['data_processing']['skewness_threshold']
            skewness = df[num_col].apply(lambda x:x.skew())

            for column in skewness[skewness>skewness_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        except Exception as e:
            logger.error("error in Data Processing step")
            raise CustomException("Failled while Data Preprocess Data",e)

    def balanceing_data(self,df):

        try:
            logger.info("Data balanching Process started")
            X=df.drop(columns='booking_status')
            y=df['booking_status']


            smote = SMOTE(random_state=42)
            x_res, y_res=smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(x_res, columns=X.columns)
            balanced_df["booking_status"] = y_res

            df = balanced_df.copy()

            logger.info("data balancing is completed")
            return df
        
        except Exception as e:
            logger.error("error in Data Balancing step")
            raise CustomException("Failled while Data Balancing Data",e)
        
    def select_features(self,df):

        try:
            logger.info("Starting Feature Selection")
           
            x = df.drop(columns='booking_status')
            y = df['booking_status']
        
            model = RandomForestClassifier(random_state=42)
            model.fit(x,y)
            
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                    'feature':x.columns ,
                    'importance':feature_importance
                })
            top_importance_features = feature_importance_df.sort_values(by='importance',ascending=False)
            
            num_feature_to_select = self.config['data_processing']['no_of_features']

            top_10_features = top_importance_features['feature'].head(num_feature_to_select).values
            logger.info(f"features selected {top_10_features}")
           
            top_10_df = df[top_10_features.tolist()+["booking_status"]]
            
            logger.info("feature selection completed successfully")
            return top_10_df


        except Exception as e:
            logger.error("error in feature selection step")
            raise CustomException("Failled while feature selection",e)
        
    
    def save_data(self,df,file_path):
        try:
            logger.info("saving our data in processed folder")

            df.to_csv(file_path,index=False)

            logger.info(f"data is saved in {file_path}")

        except Exception as e:
            logger.error("error in Data Saving step")
            raise CustomException("Failled while Data Saving",e)

    def process(self):

        try:
            logger.info("Loading data from RAW directry")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df= self.preprocess_data(test_df)

            train_df = self.balanceing_data(train_df)

            train_df = self.select_features(train_df)

            selected_columns = train_df.columns
            test_df = test_df[selected_columns]

            self.save_data(train_df,PROCESSED_TRAIN)
            self.save_data(test_df,PROCESSED_TEST)

            logger.info("data Procesess's is Finished")

        except Exception as e:
            logger.error("Error in Data Processing Steps")
            raise CustomException("FFailed in data Processing steps",e)
        

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()
