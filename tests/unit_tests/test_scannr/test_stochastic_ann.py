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
from scannr.stochastic_ann.stochastic_ffd import StochasticFFDANN


SEED = 0
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAINING_BATCH_SIZE = 2

keras.utils.set_random_seed(seed=SEED)


class DataGenerator:

    def __init__(self, case_count, feature_count, proportion) -> None:
        self.case_count = case_count
        self.feature_count = feature_count
        self.proportion = proportion
        self.raw_data = None
        self.cv_split_index = None
        self.training_data = None
        self.validation_data = None
        self.training_dataset = None
        self.validation_dataset = None

    def _generate_raw_data(self):
        """Generated data includes both features and targets, returned
        within the `tuple` object

        """
        x = np.random.randint(low=0, high=100, size=(
            self.case_count, self.feature_count
        ))
        y = np.random.gamma(shape=2, size=(self.case_count, 1))

        self.raw_data = np.hstack((x, y))

    def _compute_split_index(self):
        self.cv_split_index = floor(self.case_count*self.proportion)

    def _split_data(self) -> None:
        self.training_data = self.raw_data[0:self.cv_split_index, :]
        self.validation_data =  self.raw_data[self.cv_split_index:, ]
        
    @staticmethod
    def _generate_dataset(features:np.array,
                          targets:np.array,
                          batch_size:int) -> tf.data.Dataset:
        # Prepare the training dataset.
        return tf.data.Dataset.from_tensor_slices(
            (features, targets)
        ).batch(batch_size=batch_size)

    @staticmethod
    def get_features(array: np.array):

        return array[:, 0:-1]

    @staticmethod
    def get_targets(array: np.array):

        return array[:, -1].reshape(-1, 1)

    def _generate_datasets(self):
        self.training_dataset = self._generate_dataset(
            features=self.get_features(self.training_data),
            targets=self.get_targets(self.training_data),
            batch_size=TRAINING_BATCH_SIZE
        )
        self.validation_dataset = self._generate_dataset(
            features=self.get_features(self.validation_data),
            targets=self.get_targets(self.validation_data),
            batch_size=self.case_count-self.cv_split_index
        )

    def generate_data(self):
        """Wrapper around private methods

        """
        self._generate_raw_data()
        self._compute_split_index()
        self._split_data()
        self._generate_datasets()

class TestStochasticFFDANN(unittest.TestCase):

    def setUp(self) -> None:
        
        CASE_COUNT = 10
        FEATURE_COUNT = 5
        PROPORTION = .8
        
        self.test_data_generator = DataGenerator(
            case_count=CASE_COUNT,
            feature_count=FEATURE_COUNT,
            proportion=PROPORTION
        )
        self.test_data_generator.generate_data()

    def test_stochastic_ffdann(self):

        desired = [
            -118.65416,
            -102.33775,
            -138.45534,
            -177.433,
            -112.03105,
            -140.59764,
            -147.09407,
            -36.10840
        ]

        stochastic_ffd_ann = StochasticFFDANN(feature_total=10, k=30)
        mean, _ = \
            stochastic_ffd_ann.predict(
                self.test_data_generator.get_features(
                    self.test_data_generator.training_data
                )
            )

        np.testing.assert_almost_equal(
            desired=desired,
            actual=mean.numpy().flatten(),
            decimal=5
        )

    def tearDown(self) -> None:
        self.test_data_generator = None


def test_suite():
    
    suite = unittest.TestSuite()
    suite.addTest(TestStochasticFFDANN('test_stochastic_ffdann'))

    return suite


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(test_suite())
