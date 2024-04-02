import keras
import tensorflow as tf


class StochasticRegressionBlock(keras.layers.Layer):

    def __init__(self, feature_total):
        super().__init__()
        self.regression = keras.layers.Dense(
            units=feature_total,
            kernel_initializer='lecun_normal',
            activation='selu',
            kernel_constraint=keras.constraints.MaxNorm(2),
            bias_constraint=keras.constraints.MaxNorm(2)
        )
        self.dropout = keras.layers.AlphaDropout(rate=.5)

    def call(self, inputs):
        x = self.regression(inputs)
        x = self.dropout(x)

        return x

class StohasticFFDANN(keras.Model):

    def __init__(self, feature_total, k):
        super().__init__()
        self.k = k
        self.regression_1 = StochasticRegressionBlock(
            feature_total=feature_total
        )
        self.regression_2 = StochasticRegressionBlock(
            feature_total=feature_total
        )
        self.outputs = keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training=False):
        x = self.regression_1(inputs)
        x = self.regression_2(x)
        x = self.outputs(x)

        return x

    def predict(self, inputs):

        predictions = tf.convert_to_tensor([
            [self.call(inputs)] for _ in range(0, self.k)
        ])

        return \
            tf.math.reduce_mean(predictions, axis=0), \
            tf.math.reduce_std(predictions, axis=0)
