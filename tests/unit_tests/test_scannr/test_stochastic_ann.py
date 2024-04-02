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
        self.cv_split_index = floor(self.case_count*self.proportion)
        self.data_package = None

    def _generate_raw_data(self) -> tuple:
        """Generated data includes both features and targets, returned
        within the `tuple` object

        """
        x = np.random.randint(low=0, high=100, size=(
            self.case_count, self.feature_count
        ))
        y = np.random.gamma(shape=2, size=(self.case_count, 1))

        return x, y

    def _split_data(self, x:np.array, y:np.array) -> None:
        train_features = x[0:self.cv_split_index, :]
        train_targets = y[0:self.cv_split_index, :]
        test_features = x[self.cv_split_index:, ]
        test_targets = y[self.cv_split_index:, ]

        self.data_package = {
            'training_features': train_features, 
            'training_targets': train_targets, 
            'testing_features': test_features,
            'testing_targets': test_targets
        }

    @staticmethod
    def _generate_dataset(features:np.array,
                          targets:np.array,
                          batch_size:int) -> tf.data.Dataset:
        # Prepare the training dataset.
        return tf.data.Dataset.from_tensor_slices(
            (features, targets)
        ).batch(batch_size=batch_size)


    def generate_data(self):
        x, y = self._generate_raw_data()
        self._split_data(x=x, y=y)
        training_dataset = self._generate_dataset(
            features=self.data_package['training_features'],
            targets=self.data_package['training_targets'],
            batch_size=TRAINING_BATCH_SIZE
        )
        validation_dataset = self._generate_dataset(
            features=self.data_package['testing_features'],
            targets=self.data_package['testing_features'],
            batch_size=self.case_count-self.cv_split_index
        )

        return training_dataset, validation_dataset

class TestStochasticFFDANN(unittest.TestCase):

    def setUp(self) -> None:
        
        CASE_COUNT = 10
        FEATURE_COUNT = 5
        PROPORTION = .8
        
        test_data_generator = DataGenerator(
            case_count=CASE_COUNT,
            feature_count=FEATURE_COUNT,
            proportion=PROPORTION
        )
        self.testing_data = test_data_generator.generate_data()
        self.data_package = test_data_generator.data_package

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
            stochastic_ffd_ann.predict(self.data_package['training_features'])

        np.testing.assert_almost_equal(
            desired=desired,
            actual=mean.numpy().flatten(),
            decimal=5
        )

    def tearDown(self) -> None:
        self.testing_data = None


def test_suite():
    
    suite = unittest.TestSuite()
    suite.addTest(TestStochasticFFDANN('test_stochastic_ffdann'))

    return suite


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(test_suite())
