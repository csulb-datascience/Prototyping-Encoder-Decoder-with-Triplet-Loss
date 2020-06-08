'''
This is used to draw mesh accuracy vs gamma vs 
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM
import random
import pandas as pd


#Developed classes
import sys
sys.path.append("../classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_Autoencoder_V1 import Autoencoder
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_SVM_with_nu import SVM


#If running under eager mode ,tensorflow op will check if the inputs are 
#of type "tensorflow.python.framework.ops.EagerTensor" and keras ops are 
#implemented as DAGs. So the inputs to the eagermode will be of 
#"tensorflow.python.framework.ops.Tensor" and this throws the error
# (https://github.com/tensorflow/tensorflow/issues/34944)
tf.config.experimental_run_functions_eagerly(True)

def saveHeader(saveAs):
    #save the header
    values = [["iteration", "gamma", "nu", "tau", "lambda", "noise", "TPR", "TNR", "accuracy"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 

def save(saveAs, iteration, gamma, nu, threshold, lambdaVal, noise, results):
    #get values:
    acc1 = results[0]
    acc2 = results[1]    
    accuracy = (results[2][2] + results[3][1]) / (results[2][0] + results[3][0])
                
    values = [[iteration, gamma, nu, threshold, lambdaVal, noise, acc1, acc2, accuracy]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")  

saveAs = "./Summary.csv"
saveHeader(saveAs)

#parameters
numberPeopleTraining = 24
percentUnitsTraining = 0.7

#load the dataset
print("--> loading the dataset")
dataset = Dataset("../../data", "data_cond1_c2.csv")
print("--> dataset ready.")

#Create the noisy datasets
datasetNoise5 = Dataset("../../data", "data_cond1_c2.csv")
datasetNoise5.turnToNoisy(5)            
datasetNoise10 = Dataset("../../data", "data_cond1_c2.csv")
datasetNoise10.turnToNoisy(10)
datasetNoise20 = Dataset("../../data", "data_cond1_c2.csv")
datasetNoise20.turnToNoisy(20)
noisy=[datasetNoise5, datasetNoise10, datasetNoise20]

#Parameters only use the set of parameters that yield the best result
learningRate = 0.001
alpha = 1.0
beta = 1.0

epochs =  [30, 30, 30, 30, 30]
lambdas = [0.0, 0.1, 0.2, 0.5, 1]

lossType = 0  
batchSize = 64

thresholds = [-0.3, -0.2, -0.1, -0.05, -0.01, 0.0]
g_values_v6 = [0.001, 0.01, 0.1, 1.0, 1.5, 2.0, 2.5, 3.0]
nus = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


print("Running with new batch and nu")

    
for i in range(10):
    #prepare the dataset indexes: training, validation, unseen, test
    dataset.randomSplit(numberPeopleTraining, percentUnitsTraining)
    print("--> dataset ready.")

    print("--> Getting the datasets for training")
    x_train, y_train = dataset.getDatasetAugmented(dataset.trainingSet, multiple=batchSize)
    x_valid, y_valid = dataset.getDatasetAugmented(dataset.validationSet, multiple=batchSize)
    
    #Skip if got NAN when training
    for j, lambdaVal in enumerate(lambdas):        
        print("\n--> Iteration: ", i, " lambda: ", lambdaVal)
        
        # Creates a session 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
            
        with session.as_default():            
        
            print("--> training the CNN autoencoder")
            autoencoder_lambda = Autoencoder(dataset.unitSize(), alpha, beta, lambdaVal, learningRate, lossType)
            outerModel_lambda, encoder_lambda = autoencoder_lambda.getAutoencoderCNN()
            history = outerModel_lambda.fit(x_train, y_train, batch_size = batchSize, 
                                            validation_data = (x_valid, y_valid), epochs = epochs[j])
            lossTrain, lossValid = autoencoder_lambda.getLoss(history)
                
            #In case we got nan anyway
            if np.isnan(lossTrain) or np.isnan(lossValid): 
                print("Got NANs. skip iteration ", i)
                continue

            #Save results for encoder
            #print("--> Saving the encoder and dataset")
            encoder_lambda.save("lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_model.h5")
            dataset.saveSets(".","lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_dataset.npy")
            
            #Built the noisy datasets:
            datasetNoise5.loadSets(".","lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_dataset.npy")
            datasetNoise10.loadSets(".","lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_dataset.npy")
            datasetNoise20.loadSets(".","lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_dataset.npy")
            embeddings5 = EmbeddingsDataset(encoder_lambda, datasetNoise5)
            embeddings5.predictEmbeddings()
            embeddings10 = EmbeddingsDataset(encoder_lambda, datasetNoise10)
            embeddings10.predictEmbeddings()
            embeddings20 = EmbeddingsDataset(encoder_lambda, datasetNoise20)
            embeddings20.predictEmbeddings()
            
            #Get the random training set
            randomTrainingSet = dataset.selectRandomTrainingUnits(10)
            sizeRandomTraining, _ = dataset.dataIndexSize(randomTrainingSet)            
            embeddings_lambda = EmbeddingsDataset(encoder_lambda, dataset)
            embeddings_lambda.predictEmbeddings()

            #Train SVM with different g
            for gamma in g_values_v6:
                for nu in nus:
                    for tao in thresholds:
                        svm_lambda = SVM(embeddings_lambda)
                        svm_lambda.fit(randomTrainingSet, gamma=gamma, nu = nu)
                        
                        svm_lambda.embeddings = embeddings5
                        results = svm_lambda.accuracy(dataset.validationSet, dataset.unseenSet, tao)
                        save(saveAs, i, gamma, nu, tao, lambdaVal, 0.05, results)
                        
                        svm_lambda.embeddings = embeddings10
                        results = svm_lambda.accuracy(dataset.validationSet, dataset.unseenSet, tao)
                        save(saveAs, i, gamma, nu, tao, lambdaVal, 0.10, results)
                        
                        svm_lambda.embeddings = embeddings20
                        results = svm_lambda.accuracy(dataset.validationSet, dataset.unseenSet, tao)
                        save(saveAs, i, gamma, nu, tao, lambdaVal, 0.20, results)
                                         
            #save the embeddings dataset
            #print("--> saving the embeddings")
            embeddings_lambda.save(".", "lambda_" + str(lambdaVal) + "_iter_" + str(i) + "_embeddings.npy")
            #print("  ... Embeddings Saved")
            
        session.close()            






