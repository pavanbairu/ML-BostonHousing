import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.exception import BostonHousingException
from src import constants
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    # Configuration for data transformation paths
    scaled_trained_path = constants.SCALED_TRAINED_PATH
    scaled_test_path = constants.SCALED_TEST_PATH
    preproccessor_path = constants.PREPROCESSOR_PATH

class DataTransformation:

    def __init__(self):
        # Initialize DataTransformation with configuration
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Performs data transformation on the training and testing datasets.
        This includes scaling the features and preparing the final arrays for model training.
        
        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.
        
        Returns:
            tuple: Transformed training and testing arrays.
        
        Raises:
            BostonHousingException: If any error occurs during data transformation.
        """
        try:
            logging.info("Data transformation initiated")
            
            # Read the training and testing datasets
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Read the train and test dataset")

            target_feature = "MEDV"  # Define the target feature

            # Separate features and target variable from training and testing data
            input_train_data = train_data.drop(target_feature, axis=1)
            input_test_data = test_data.drop(target_feature, axis=1)
            logging.info("Removed target feature from train and test dataset for transform input data")

            # columns = input_train_data.columns  # Get the feature columns
            num_columns = [column for column in input_train_data.columns if input_train_data[column].nunique() > 25]
            cat_column = ['ZN']
            num_pieline = Pipeline(
                steps=[

                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Create a ColumnTransformer for scaling the features
            pre_processor = ColumnTransformer([
                ('numpipeline', num_pieline, num_columns),  # Apply StandardScaler to all feature columns
                ('catpipeline', cat_pipeline, cat_column)
            ])

            # Fit the preprocessor on the training data and transform both train and test data
            scaled_train_data = pre_processor.fit_transform(input_train_data)
            scaled_test_data = pre_processor.transform(input_test_data)
            logging.info("Scaled the train and test data")

            # Combine scaled features with the target variable to create final arrays
            train_arr = np.c_[scaled_train_data, np.array(train_data[target_feature])]
            test_arr = np.c_[scaled_test_data, np.array(test_data[target_feature])]

            # Save the preprocessor object for later use
            save_obj(self.data_transformation_config.preproccessor_path, pre_processor)
            
            return (train_arr, test_arr)  # Return the transformed arrays

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise BostonHousingException(e, sys)