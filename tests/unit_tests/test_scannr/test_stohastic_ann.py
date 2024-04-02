from scannr.stohastic_ann.stohastic_ffd import StohasticFFDANN
import unittest
import numpy as np
import keras
import tensorflow as tf
from math import floor


def generate_raw_data(case_count, feature_count) -> tuple:
    """Generated data includes both features and targets, returned
    within the `tuple` object.

    """
    x = np.random.randint(low=0, high=100, size=(
        case_count, feature_count
    ))
    y = np.random.gamma(shape=2, size=(case_count, 1))

    return x, y


def split_data(proportion:float, 
            case_total:int,
            x:np.array,
            y:np.array) -> tuple:
    cv_split_index = floor(case_total*proportion)  # Safe rounding.
    train_features = x[0:cv_split_index, :]
    train_targets = y[0:cv_split_index, :]
    test_features = x[cv_split_index:, ]
    test_targets = y[cv_split_index:, ]

    return train_features, train_targets, test_features, \
        test_targets


def prepare_data(train_features:np.array,
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

    def test_stochastic_ffdann(self):

        SEED = 0
        CASE_COUNT = 100
        FEATURE_COUNT = 10
        PROPORTION = .8  # Proportion of data to use for CV.
        LEARNING_RATE = 1e-3
        EPOCHS = 10
        TRAINING_BATCH_SIZE = 2

        keras.utils.set_random_seed(seed=SEED)

        # Generate features and targets.  
        features, targets = generate_raw_data(
            case_count=CASE_COUNT, feature_count=FEATURE_COUNT
        )

        # Split raw data.
        training_features, training_targets, testing_features, testing_targets = \
            split_data(
                proportion=PROPORTION,
                case_total=CASE_COUNT,
                x=features,
                y=targets
            )
        
        # Administer raw data via `tf.data.Dataset` class.
        training_dataset, validation_dataset = prepare_data(
            train_features=training_features,
            train_targets=training_targets,
            test_features=testing_features,
            test_targets=testing_targets,
            training_batch_size=TRAINING_BATCH_SIZE,
            validation_batch_size=CASE_COUNT-floor(CASE_COUNT*PROPORTION)
        )

        stochastic_ffd_ann = StohasticFFDANN(feature_total=10, k=30)
        stochastic_ffd_ann(training_features)

        print('Hello Wild')

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestStochasticFFDANN('test_stochastic_ffdann'))

    return suite


if __name__ == '__main__':

    runner = unittest.TextTestRunner()
    runner.run(test_suite())
