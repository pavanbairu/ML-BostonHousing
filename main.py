import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.exception import BostonHousingException



if __name__ == "__main__":

    try:
        # Main execution block
        # Create an instance of DataIngestion
        data_ingestion = DataIngestion()
        
        # Initiate data ingestion and retrieve paths for train and test data
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Create an instance of DataTransformation to transform the data
        data_transformation = DataTransformation()
        
        # Initiate data transformation and retrieve transformed training and testing arrays
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)

        # Create an instance of ModelTraining to train the model
        model_trainer = ModelTraining()
        
        # Initiate model training with the transformed training and testing data
        model_trainer.initiate_model_training(train_arr, test_arr)

    except Exception as e:
        raise BostonHousingException(e, sys)