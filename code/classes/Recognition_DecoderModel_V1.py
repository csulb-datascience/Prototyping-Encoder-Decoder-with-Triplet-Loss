# Project: Prototyping-Encoder-Decoder-with-Triplet-Loss
# version: 1.0
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
# Date: June 2020
#
#Include a reference to this site if you will use this code.

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Recognition_Conv1DTranspose_V1 import Conv1DTranspose

class DecoderModel:
    def __init__(self):
        self.embeddingSize = 128
        self.numModal = 3
    
    #Decode the individual CNN 
    def getSingleCNN(self, unitSize, features):
        #Reshape each modal to flow on its own cnn
        netInput = Input(shape= unitSize * 128)
        x = Reshape((unitSize, netInput.shape[1]//unitSize))(netInput)
        x = BatchNormalization()(x)
    
        # Reverse the convolution
        x = Conv1DTranspose(128, kernel_size=20, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)    
        
        # Reverse the convolution
        x = Conv1DTranspose(64, kernel_size=20, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Reverse the convolution
        x = Conv1DTranspose(32, kernel_size=20, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    
        #Back to the original shape
        x = Conv1D(features, kernel_size=20, activation='tanh', padding='same')(x)
        model = Model(inputs=netInput, outputs=x)                  
        return (model)
    
    
    #Decoder of the multimodal network
    def getDecoderCNN(self, unitSize):            

        #base layer -> reverse the encoder
        embedding = Input(shape= self.embeddingSize)
        x = Dense(256)(embedding)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
            
        #reverse the size of the concatenation
        x = Dense(unitSize * self.numModal * self.embeddingSize)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    
        #Split the tensor for each modal
        press, acc, gyro = tf.split(x, num_or_size_splits=self.numModal, axis=1) 
        
        #Obtain the unit
        press = self.getSingleCNN(unitSize, 16)(press)
        acc = self.getSingleCNN(unitSize, 6)(acc)
        gyro = self.getSingleCNN(unitSize, 6)(gyro)
        unit = concatenate([press, acc, gyro], axis=2)
        
        #Return the model
        model = Model(inputs=embedding, outputs=unit)
        return(model)
       

    #Returns the compiled Decoder model
    def getCompiledCNN(self, unitSize):
        model = self.getDecoderCNN(unitSize)
        model.compile(loss='mean_squared_error', optimizer = Adam(), metrics=['accuracy'])
        return(model)    

    
    
