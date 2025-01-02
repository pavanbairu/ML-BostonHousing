import os
import sys
import numpy as np
import pandas as pd


TRAIN_DATA_PATH = os.path.join("Artifacts", "train.csv")
TEST_DATA_PATH = os.path.join("Artifacts", "test.csv")
ORIGINAL_DATA_PATH = os.path.join("Artifacts", "data.csv")
TEST_SIZE=0.3
RANDOM_STATE=42

SCALED_TRAINED_PATH = os.path.join(os.getcwd(), "Artifacts", "scaled_train_data.csv")
SCALED_TEST_PATH= os.path.join(os.getcwd(), "Artifacts", "scaled_test_data.csv")
PREPROCESSOR_PATH = os.path.join(os.getcwd(), "Artifacts", "preprocessor.pkl")

MODEL_PATH = os.path.join(os.getcwd(), "Artifacts", "model.pkl")
REGULARIZATION_STRENGTH = 1e-10
