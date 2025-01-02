import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from housing.exception import BostonHousingException
from housing.logger import logging
from housing.components.data_transformation import DataTransformation
from housing.components.model_training import ModelTraining
from housing import constants

@dataclass
class DataIngestionConfig:
    # Configuration for data ingestion paths
    train_data_path = constants.TRAIN_DATA_PATH
    test_data_path = constants.TEST_DATA_PATH
    original_data_path = constants.ORIGINAL_DATA_PATH
    test_size=constants.TEST_SIZE
    random_state=constants.RANDOM_STATE


class DataIngestion:
    def __init__(self):
        # Initialize DataIngestion with configuration
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the cleaned dataset, splits it into training and testing sets,
        and saves these sets to specified paths.
        
        Returns:
            tuple: Paths to the training and testing CSV files.
        
        Raises:
            BostonHousingException: If any error occurs during data ingestion.
        """
        try:
            # Read the cleaned dataset from the specified path
            df = pd.read_csv(f"{os.path.join(os.getcwd())}/datasets/BostonHousingCleaned.csv")
            logging.info("Read the cleaned dataset")
 
            # Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, 
                                                   test_size=self.data_ingestion_config.test_size,
                                                    random_state=self.data_ingestion_config.random_state
                                                )
            logging.info("Performed train-test split operation")

            # Create the directory for saving the train and test files if it doesn't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            # Save the training and testing sets to CSV files
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            df.to_csv(self.data_ingestion_config.original_data_path, index=False, header=True)
            logging.info("Saved the train and test files")        

            return (self.data_ingestion_config.train_data_path, 
                    self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise BostonHousingException(e, sys)
