import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np
 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str =os.path.join('artifacts',"train.csv")
    test_data_path: str =os.path.join('artifacts',"test.csv")
    raw_data_path: str =os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join("Data", "Base_Dataset.csv"))

            logging.info('Read the Dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header = True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header = True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header = True)

            logging.info("ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    config = DataTransformationConfig()
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(train_path, test_path)

 
    trainer = ModelTrainer(n_estimators=1000, recall_target=0.98)
    trainer.train(X_train, y_train, X_test, y_test)
    trainer.evaluate(X_test, y_test)
    trainer.save_model("lightgbm_model.pkl")


    # Later load it
    # trainer.load_model("lightgbm_model.pkl")
    # preds = trainer.predict(X_test)

