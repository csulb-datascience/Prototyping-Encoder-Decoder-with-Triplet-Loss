import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Recognition_TripletLoss_V1 import combined_loss
from Recognition_DecoderModel_V1 import DecoderModel
from Recognition_EncoderModel_V1 import EncoderModel

class Autoencoder:
    #constructor
    def __init__(self, unitSize, alpha=1.2, beta=1.0, lamda=0.0001, learningRate=0.001, lossType=0):
        self.unitSize = unitSize
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.learningRate = learningRate
        self.lossType = lossType        
        self.encoderModel = EncoderModel()
        self.decoderModel = DecoderModel()                
                                
    #Just return the last value of y_pred 
    def bridge_loss(self, y_true, y_pred):
        return(y_pred)

    #returns loss = triplet_loss + lamda * DAE_loss
    def combinedLoss(self, x):
        y_true = x[0]
        y_pred = x[1]
        decoded = x[2]
        c_mean = x[3]        
        loss = combined_loss(y_true, y_pred, decoded, c_mean, self.alpha, self.beta, self.lamda, self.lossType)
        return(loss)

    #Dummy Model just to get features as input
    def getFeature(self, dimension):
        feature = Input(shape=dimension)
        model = Model (feature, feature)
        return(model)           
    
    #Build the autoencoder
    def getAutoencoderCNN(self):
        #Insert the encoder
        encoder = self.encoderModel.getEncoderCNN(self.unitSize)
        embeddings = encoder.output

        #Insert the decoder
        decoder = self.decoderModel.getDecoderCNN(self.unitSize)
        decodedUnits = decoder(embeddings)
        
        #Calculate the combined loss
        y_true = self.getFeature(dimension=1)
        unitMean = self.getFeature(dimension=(self.unitSize, 28))
        loss =  Lambda(self.combinedLoss)([y_true.output, embeddings, decodedUnits, unitMean.output])
                            
        #Set and compile the model for the Loss calculation
        inputModel = [[encoder.input], y_true.input, unitMean.input]
        autoencoder = Model(inputs= inputModel, outputs=loss)
        autoencoder.compile(optimizer=Adam(self.learningRate), loss=self.bridge_loss)
        
        #Return the outer model for fitting and the encoder
        return(autoencoder, encoder)
    
    #Return the last loss from the history
    def getLoss(self, history):
        #get the values
        lossTrain = history.history["loss"][-1]
        lossValid = history.history["val_loss"][-1]        
        return(lossTrain, lossValid)    
