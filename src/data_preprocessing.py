import pandas as pd
import numpy as np
from constants import PATH_TO_TRAIN_DATA, PATH_TO_TEST_DATA


class DataPreprocessor():

    def __init__(self, dataset='train'):
        if dataset == 'train':
            self.data = self.load_raw_data(PATH_TO_TRAIN_DATA)
        elif dataset == 'test':
            self.data = self.load_raw_data(PATH_TO_TEST_DATA)

        self.dataset = dataset
        self.digits = None
        self.labels = None
        self.preprocessing_done = False

    @staticmethod
    def load_raw_data(path_to_data):
        return pd.read_csv(path_to_data)

    def preprocess(self):
        if self.dataset == 'train':
            self.split_digits_and_labels()
        elif self.dataset == 'test':
            self.digits = self.data.values

        self.reshape_digits()
        self.rescale_digits()
        self.preprocessing_done = True

    def split_digits_and_labels(self):
        self.digits = self.data.loc[:, 'pixel0':].values
        self.labels = self.data.loc[:, 'label'].values

    def reshape_digits(self):
        self.digits = self.digits.reshape(self.digits.shape[0], 28, 28, 1)

    def rescale_digits(self):
        self.digits = self.digits / np.max(self.digits)

    def get_digits(self):
        assert self.preprocessing_done
        return self.digits

    def get_labels(self):
        assert self.preprocessing_done
        return self.labels