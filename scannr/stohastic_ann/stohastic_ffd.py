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

    def call(self, input_tensor, training):
        x = self.regression(input_tensor)
        x = self.dropout(x, training=training)

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
        x = self.regression_1(training=training)(inputs)
        x = self.regression_2(x, training=training)
        x = self.outputs(x)

        # # Custom prediction.
        # predictions = [
        #     [self.outputs(x)] for _ in range(0, self.k)
        # ]

        # return \
        #     tf.reduce_mean(tf.convert_to_tensor(predictions), axis=0)
