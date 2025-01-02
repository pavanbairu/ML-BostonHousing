import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataclasses import dataclass
from src.exception import BostonHousingException
from src import constants
from src.utils import save_obj
from sklearn.linear_model import LinearRegression
from src.logger import logging
import statsmodels.api as sm

@dataclass
class ModelTrainingConfig:
    # Configuration for model training paths
    model_path = constants.MODEL_PATH
     # Regularization strength to prevent overfitting
    regularization_strength = constants.REGULARIZATION_STRENGTH

class ModelTraining:

    def __init__(self):
        # Initialize ModelTraining with configuration
        self.model_training_config = ModelTrainingConfig()
    
    def lr(self, X_train, y_train):
        """
        Performs linear regression using the normal equation method.
        
        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
        
        Returns:
            dict: A dictionary containing the intercept and coefficients of the model.
        
        Raises:
            BostonHousingException: If any error occurs during model training.
        """
        # Add intercept term to the feature data
        X_train = np.insert(X_train, 0, 1, axis=1)

        try:
                   
            # Calculate the coefficients using the normal equation with regularization
            self.betas = np.linalg.inv(np.dot(X_train.T, X_train) + 
                                       self.model_training_config.regularization_strength * np.eye(X_train.shape[1])).dot(X_train.T).dot(y_train)
            logging.info("Calculated the slope and intercept")
            
            # Extract intercept and coefficients
            self.intercept_ = self.betas[0]
            self.coef_ = self.betas[1:]    

            return {
                "intercept": self.intercept_, 
                "coefficients": self.coef_
            }

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise BostonHousingException(e, sys)
        

    def initiate_model_training(self, train_data, test_data):
        """
        Initiates the model training process, including training a linear regression model
        and evaluating its performance on test data.
        
        Args:
            train_data (np.ndarray): The training data array.
            test_data (np.ndarray): The testing data array.
        
        Raises:
            BostonHousingException: If any error occurs during model training.
        """
        try:
            logging.info("Initiated model training")
            
            # Separate features and target variable from training and testing data
            X_train, y_train, X_test, y_test = (
                train_data[:, :-1],
                train_data[:, -1],
                test_data[:, :-1],
                test_data[:, -1]
            )

            logging.info("Separated X_train, X_test, y_train, and y_test from train and test array of data")

            # Train the scikit-learn Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions on the test data
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics for the scikit-learn model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            score = model.score(X_train, y_train)

            print("*" * 50)
            print("scikit-learn model")
            print("---------------------------")
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"R-squared: {r2}")
            print(f"score: ", score)

            print("*" * 50)
            print("scratch model")
            print("---------------------------")
            
            # Train the scratch model using the custom linear regression method
            self.model = self.lr(X_train, y_train)
            logging.info("Built the model")
            
            # Save the model object for later use
            save_obj(self.model_training_config.model_path, self.model)

            # Add intercept term to the test data for scratch model predictions
            X_test = np.insert(X_test, 0, 1, axis=1)

            # Make predictions on the test data using the scratch model
            y_pred = np.dot(X_test, self.betas)
            logging.info("Performed prediction on test data")

            # Calculate evaluation metrics for the scratch model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)


            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"R-squared: {r2}")


            return "y predictions are :", y_pred

        except Exception as e:
            raise BostonHousingException(e, sys)