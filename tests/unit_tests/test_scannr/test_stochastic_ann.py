from scannr.stochastic_ann.stochastic_ffd import StochasticFFDANN
import unittest
import time
import os
from math import floor
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras


class TestDataGenerator:

    def __init__(self, case_count, feature_count, proportion) -> None:
        self.case_count = case_count
        self.feature_count = feature_count

    def _generate_raw_data(self, ) -> tuple:
        """Generated data includes both features and targets, returned
        within the `tuple` object.

        """
        x = np.random.randint(low=0, high=100, size=(
            self.case_count, self.feature_count
        ))
        y = np.random.gamma(shape=2, size=(self.case_count, 1))

        return x, y
    
    def _compute_split_index(self):
        self.cv_split_index = floor(self.case_total*self.proportion)

    def _split_data(self,x:np.array, y:np.array) -> tuple:
        train_features = x[0:self.cv_split_index, :]
        train_targets = y[0:self.cv_split_index, :]
        test_features = x[self.cv_split_index:, ]
        test_targets = y[self.cv_split_index:, ]

        return train_features, train_targets, test_features, \
            test_targets


    def generate_data(self,
                      train_features:np.array,
                      train_targets:np.array,
                      training_batch_size:int,
                      test_features:np.array,
                      test_targets:np.array,
                      validation_batch_size:int) -> tf.data.Dataset:
        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_targets)
        ).batch(batch_size=training_batch_size)

        # Prepare the validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (test_features, test_targets)
        ).batch(batch_size=validation_batch_size)

        return train_dataset, val_dataset


class TestStochasticFFDANN(unittest.TestCase):
    
    def setUp(self) -> None:
        test_data_generator = TestDataGenerator()
        self.testing_data = test_data_generator.generate_data()

    def test_stochastic_ffdann(self):

        stochastic_ffd_ann = StochasticFFDANN(feature_total=10, k=30)
        mean, variance = stochastic_ffd_ann.predict(training_features)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestStochasticFFDANN('test_stochastic_ffdann'))

    return suite


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(test_suite())
