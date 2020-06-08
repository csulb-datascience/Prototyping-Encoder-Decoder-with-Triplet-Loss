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
from Recognition_TripletLoss_V1 import custom_loss

class EncoderModel:
    #constructor
    def __init__(self):
        self.embeddingSize = 128
                
    #Load a model from disk in h5 format
    def loadModel(self, fileName):
        model = tf.keras.models.load_model(fileName) #, compile=False)
        return(model)
        
    #****************************************************************************************************
    # CNN
    #****************************************************************************************************
        
    #Network used for each sensor individualy
    def getCNN(self, height, width):
        #The input is a unit step. 
        input_shape = (height, width)
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=20, padding='Same', activation="relu", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv1D(filters = 64, kernel_size=20, padding = 'Same', activation ='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters = 128, kernel_size=20, padding = "Same", activation = "relu"))
        model.add(BatchNormalization())
        model.add(Flatten())       
        return(model)
    
    #Merge the networks of each sensor in a unique network
    def getEncoderCNN(self, unitSize):
        CNN_press = self.getCNN(height=unitSize, width=16)
        CNN_acc = self.getCNN(height=unitSize, width=6)
        CNN_gyro = self.getCNN(height=unitSize, width=6)
    
        # Combine the outputs of the CNNs and complete other layers
        combinedInput = concatenate([CNN_press.output, CNN_acc.output, CNN_gyro.output])
        x = Dense(256, activation="relu")(combinedInput)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)    
        
        #normalize embedding
        x = Dense(self.embeddingSize)(x)        
        embeddings = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        
        # Return the Model
        model = Model(inputs=[CNN_press.input, CNN_acc.input, CNN_gyro.input], outputs=embeddings)
        return(model)
    
    #Return the compiled model using the triplet loss function
    def getCompiledCNN(self, unitSize, alpha, beta , learningRate):
        model = self.getEncoderCNN(unitSize)
        model.compile( optimizer=Adam(learningRate), loss=custom_loss(alpha, beta))
        return(model)
    
    
    #****************************************************************************************************
    # LSTM
    #****************************************************************************************************

    def getLSTM(self, timeSteps, dataDim):
        #Build the model as a sequence of layers
        model = Sequential()    
        hidden_cells = 64
        model.add(LSTM(hidden_cells, input_shape=(timeSteps, dataDim), return_sequences=True, recurrent_dropout=0.2)) 
        model.add(LSTM(hidden_cells))    #A second layer of LSTM
        model.add(Flatten())
        return (model)

    def getEncoderLSTM(self, unitSize):
        LSTM_press = self.getLSTM(timeSteps=unitSize, dataDim=16)
        LSTM_acc = self.getLSTM(timeSteps=unitSize, dataDim=6)
        LSTM_gyro = self.getLSTM(timeSteps=unitSize, dataDim=6)

        # Combine the outputs of the LSTMs and complete other layers
        combinedInput = concatenate([LSTM_press.output, LSTM_acc.output, LSTM_gyro.output])
        x = Dense(256, activation="relu")(combinedInput)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        #Embedding
        x = Dense(self.embeddingSize)(x)    
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    
        # Return the final Model
        model = Model(inputs=[LSTM_press.input, LSTM_acc.input, LSTM_gyro.input], outputs=x)
        return model
        
    def getCompiledLSTM(self, unitSize, alpha, beta, learningRate):
        # Return the final Model
        model = self.getEncoderLSTM(unitSize)
        model.compile( optimizer=Adam(learningRate), loss=custom_loss(alpha, beta))   
        #print(model.summary)
        return(model)    
    
    def getLoss(self, history):
        #get the values
        lossTrain = history.history["loss"][-1]
        lossValid = history.history["val_loss"][-1]        
        return(lossTrain, lossValid)    
    