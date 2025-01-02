import os
import sys
from housing.exception import BostonHousingException
from housing.logger import logging
import pickle

def save_obj(path, obj):
    """
    Saves a Python object to a specified file path using pickle.
    
    Args:
        path (str): The file path where the object will be saved.
        obj (any): The Python object to be saved.
    
    Raises:
        BostonHousingException: If any error occurs during the saving process.
    """
    try:
        # Create the directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Open the file in write-binary mode and save the object
        with open(path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise BostonHousingException(e, sys)
