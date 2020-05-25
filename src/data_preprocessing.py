import pandas as pd
import numpy as np
from constants import PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA


def load_raw_data(path_to_data):
    return pd.read_csv(path_to_data)

def split_digits_and_labels(raw_data):
    digits = raw_data.loc[:, 'pixel0':]
    labels = raw_data.loc[:, 'label']
    return digits, labels

def reshape_digits(digits):
    return digits.reshape(digits.shape[0], 28, 28, 1)

def rescale_digits(digits):
    return digits / np.max(digits)

def get_train_data():
    raw_data = load_raw_data(PATH_TO_TRAIN_DATA)
    digits, labels = split_digits_and_labels(raw_data)

    # Pandas dataframe to numpy
    digits = digits.values
    labels = labels.values

    digits = reshape_digits(digits)
    digits = rescale_digits(digits)
    return digits, labels

def get_test_data():
    raw_data = load_raw_data(PATH_TO_TEST_DATA)

    # Pandas dataframe to numpy
    test_digits = raw_data.values

    test_digits = reshape_digits(test_digits)
    test_digits = rescale_digits(test_digits)
    return test_digits