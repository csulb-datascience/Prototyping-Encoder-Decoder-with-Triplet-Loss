import tensorflow as tf


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.transpose2D = tf.keras.layers.Conv2DTranspose(filters,
                    (kernel_size, 1), (strides, 1), padding)        

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.transpose2D(x)
        x = tf.squeeze(x, axis=2)
        return x
     