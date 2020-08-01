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
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Recognition_TripletLoss_V2 import TripletSemiHardLoss

class EncoderModel:
    #constructor
    def __init__(self):
        self.embeddingSize = 128
        self.featuresPress = 16
        self.featuresAcc = 6
        self.featuresGyro = 6        
                
    #Load a model from disk in h5 format
    def loadModel(self, fileName):
        #model = tf.keras.models.load_model(fileName, compile=False)        
        model = tf.keras.models.load_model(fileName, custom_objects={'TripletSemiHardLoss':TripletSemiHardLoss})
        return(model)
        
    #****************************************************************************************************
    # CNN
    #****************************************************************************************************
        
    #Network used for each sensor individualy
    def getBranchCNN(self, inputs):
        x = Conv1D(filters=32, kernel_size=20, padding='Same', activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters = 64, kernel_size=20, padding = 'Same', activation ='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters = 128, kernel_size=20, padding = "Same", activation = "relu")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        return(x)
    
    #Merge the branchs of each sensor in a unique network
    def getNetworkEncoderCNN(self, inputs):
        CNN_press = self.getBranchCNN(inputs[0])
        CNN_acc = self.getBranchCNN(inputs[1])
        CNN_gyro = self.getBranchCNN(inputs[2])
    
        # Combine the outputs of the CNNs and complete other layers
        combinedOutput = concatenate([CNN_press, CNN_acc, CNN_gyro])
        x = Dense(256, activation="relu", name="dense_combined")(combinedOutput)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)    
        
        #normalize embedding
        x = Dense(self.embeddingSize, name="embedding")(x)        
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="encoder")(x)        
        return(x)
        
    #Returns the model of the encoder
    def getEncoderCNN(self, unitSize):
        inputPress = Input(shape = (unitSize, self.featuresPress), name="input_press")
        inputAcc = Input(shape = (unitSize, self.featuresAcc), name="input_acc")
        inputGyro = Input(shape = (unitSize, self.featuresGyro), name="input_gyro")
        inputs = [inputPress, inputAcc, inputGyro]
        embeddings = self.getNetworkEncoderCNN(inputs)        
        model = Model(inputs=inputs, outputs=embeddings)
        return(model)
    
    #Return the compiled model using the triplet loss function
    def getCompiledCNN(self, unitSize, alpha, beta , learningRate):
        model = self.getEncoderCNN(unitSize)
        model.compile( optimizer=Adam(learningRate), loss=TripletSemiHardLoss(alpha, beta))
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
        