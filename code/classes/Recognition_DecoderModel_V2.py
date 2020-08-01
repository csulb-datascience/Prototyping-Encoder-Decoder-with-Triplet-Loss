# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#
# Class version: 1.0
# Date: July 2020
#
# Include a reference to this site if you will use this code.

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Recognition_Conv1DTranspose_V1 import Conv1DTranspose
from Recognition_PrototypeLoss_V1 import PrototypeLoss

class DecoderModel:
    def __init__(self, embeddingSize):
        self.embeddingSize = embeddingSize
        self.numModal = 3
        self.featuresPress = 16
        self.featuresAcc = 6
        self.featuresGyro = 6
    
    #Decoder for a sensor branch
    def getBranchCNN(self, inputs, unitSize, features):
        #Unflatten the input to a 2D shape
        x = Reshape((unitSize, inputs.shape[1]//unitSize))(inputs)
        x = BatchNormalization()(x)
    
        # Reverse the convolution
        x = Conv1DTranspose(128, kernel_size=20, padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        
        # Reverse the convolution
        x = Conv1DTranspose(64, kernel_size=20, padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        
        # Reverse the convolution
        x = Conv1DTranspose(32, kernel_size=20, padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
    
        #Back to the original shape
        x = Conv1D(features, kernel_size=20, activation='tanh', padding='same')(x)
        return (x)
    
    
    #Decoder of the multimodal network
    def getNetworkDecoderCNN(self, inputs, unitSize):
        #base layer -> reverse the encoder
        x = Dense(256, activation="relu")(inputs)
        x = BatchNormalization()(x)
            
        #reverse the size of the concatenation
        x = Dense(unitSize * self.numModal * self.embeddingSize, activation="relu")(x)
        x = BatchNormalization()(x)
    
        #Split the tensor for each modal
        press, acc, gyro = tf.split(x, num_or_size_splits=self.numModal, axis=1) 
        
        #Obtain the unit
        press = self.getBranchCNN(press, unitSize, self.featuresPress)
        acc = self.getBranchCNN(acc, unitSize, self.featuresAcc)
        gyro = self.getBranchCNN(gyro, unitSize, self.featuresGyro)        
        unit = concatenate([press, acc, gyro], axis=2, name="unit_step")
        unit = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="decoder")(unit)                
        return(unit)       
    
    #Returns the model of the decoder
    def getDecoderCNN(self, unitSize):            
        #base layer -> reverse the encoder
        inputs = Input(shape= self.embeddingSize, name="input_decoder")
        unit = self.getNetworkDecoderCNN(inputs, unitSize)
        model = Model(inputs=inputs, outputs=unit)
        return(model)
          
    #Returns the compiled Decoder model
    def getCompiledCNN(self, unitSize):
        model = self.getDecoderCNN(unitSize)
        #model.compile(loss='mean_squared_error', optimizer = Adam(), metrics=['accuracy'])
        model.compile(loss=PrototypeLoss(), optimizer = Adam(), metrics=['accuracy'])
        return(model)    

    #****************************************************************************************************
    # OTHERS
    #****************************************************************************************************
    
    def getResults(self, history):
        #get the values
        lossTrain, lossValid = -1, -1
        if "loss" in history.history.keys(): lossTrain = history.history["loss"][-1]
        if "val_loss" in history.history.keys(): lossValid = history.history["val_loss"][-1]        
        return(lossTrain, lossValid)    
            