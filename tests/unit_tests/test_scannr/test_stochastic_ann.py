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


SEED = 0
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAINING_BATCH_SIZE = 2

keras.utils.set_random_seed(seed=SEED)


class TestDataGenerator:

    def __init__(self, case_count, feature_count, proportion) -> None:
        self.case_count = case_count
        self.feature_count = feature_count
        self.proportion = proportion
        self.cv_split_index = floor(self.case_count*self.proportion)

    def _generate_raw_data(self) -> tuple:
        """Generated data includes both features and targets, returned
        within the `tuple` object.

        """
        x = np.random.randint(low=0, high=100, size=(
            self.case_count, self.feature_count
        ))
        y = np.random.gamma(shape=2, size=(self.case_count, 1))

        return x, y

    def _split_data(self, x:np.array, y:np.array) -> tuple:
        train_features = x[0:self.cv_split_index, :]
        train_targets = y[0:self.cv_split_index, :]
        test_features = x[self.cv_split_index:, ]
        test_targets = y[self.cv_split_index:, ]

        return train_features, train_targets, test_features, \
            test_targets

    @staticmethod
    def _generate_data(features:np.array,
                       targets:np.array,
                       batch_size:int) -> tf.data.Dataset:
        # Prepare the training dataset.
        dataset = tf.data.Dataset.from_tensor_slices(
            (features, targets)
        ).batch(batch_size=batch_size)

        return dataset

    def generate_data(self):
        x, y = self._generate_raw_data()
        training_features, training_targets, testing_features, \
            testing_targets = self._split_data(x=x, y=y)
        training_dataset = self._generate_data(
            features=training_features,
            targets=training_targets,
            batch_size=2
        )
        validation_dataset = self._generate_data(
            features=testing_features,
            targets=testing_targets,
            batch_size=self.case_count-self.cv_split_index
        )

        return training_dataset, validation_dataset

class TestStochasticFFDANN(unittest.TestCase):

    def setUp(self) -> None:
        CASE_COUNT = 10
        FEATURE_COUNT = 5
        PROPORTION = .8
        test_data_generator = TestDataGenerator(
            case_count=CASE_COUNT,
            feature_count=FEATURE_COUNT,
            proportion=PROPORTION
        )
        self.testing_data = test_data_generator.generate_data()

    def test_stochastic_ffdann(self):

        stochastic_ffd_ann = StochasticFFDANN(feature_total=10, k=30)
        mean, variance = stochastic_ffd_ann.predict(self.testing_data[0])

    def tearDown(self) -> None:
        self.testing_data = None


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestStochasticFFDANN('test_stochastic_ffdann'))

    return suite


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(test_suite())
