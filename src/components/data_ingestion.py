import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str:os.path.join('artifact', "train.csv")
    test_data_path: str:os.path.join('artifact', "test.csv")
    raw_data_path: str:os.path.join('artifact', "data.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv("model_development/training_data/train.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            
            
            
    





