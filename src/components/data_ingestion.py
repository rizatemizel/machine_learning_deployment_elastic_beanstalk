import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation_and_training_pipeline import PipelinePathConfig, PipelineTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df=pd.read_csv('model_development\\training_data\\data.csv')
            logging.info('Reading data set')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            # Drop outliers
            df = df.drop(df[(df['GrLivArea'] > 4000)
                                                & (df['SalePrice'] < 200000)].index)
            df = df.drop(df[(df['GarageArea'] > 1200)
                                                & (df['SalePrice'] < 300000)].index)
            df = df.drop(df[(df['TotalBsmtSF'] > 4000)
                                                & (df['SalePrice'] < 200000)].index)
            df = df.drop(df[(df['1stFlrSF'] > 4000)
                                                & (df['SalePrice'] < 200000)].index)

            df = df.drop(df[(df['TotRmsAbvGrd'] > 12)
                                                & (df['SalePrice'] < 230000)].index)




            logging.info("Train test split is initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    
    trainer = PipelineTrainer()
    trainer.train_pipeline(train_data_path,test_data_path)