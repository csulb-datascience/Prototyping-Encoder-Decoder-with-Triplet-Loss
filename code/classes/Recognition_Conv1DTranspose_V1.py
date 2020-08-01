# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Class version: 1.0
# Date: June 2020
#
# Include a reference to this site if you will use this code.


import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose

class Conv1DTranspose(tf.keras.layers.Layer):
    
    #Input parameters
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, **kwargs):
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation        
        
    # Creates the state of the layer. 
    def build(self, input_shape):  
        self.transpose1D = Conv2DTranspose(filters = self.filters, 
                                           kernel_size = (self.kernel_size,1), 
                                           strides = (self.strides,1), 
                                           padding = self.padding, 
                                           activation=self.activation)        
        
    # performs the logic of applying the layer to the input tensors 
    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        #x = Lambda(lambda x: tf.expand_dims(x, axis=2))(x)
        x = self.transpose1D(x)
        x = tf.squeeze(x, axis=2)
        #x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
        return x    
    
    #Returns a dictionary containing the configuration used to initialize this layer
    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
        })
        return config                
