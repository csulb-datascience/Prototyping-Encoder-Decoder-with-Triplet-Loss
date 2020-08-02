# Prototyping-Encoder-Decoder-with-Triplet-Loss
#
# Program to train the models.
#   the results are saved in a csv file "summary.csv"
#
# We repeat our experiment 20 times. Each time, we select 80% of 30 participants randomly. 
# For each participant in selected 24 people, 75% of unit steps are allocated to the training 
# set and the rest is assigned to the known test data set. 
# In addition, for the remaning 20% of 30 participants(6 people), all unit steps belong to 
# unknown test data set.
#
# version: 2.0
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
# Date: July 2020
#
#Include a reference to this site if you will use this code.

import tensorflow as tf
import pandas as pd

#classes
import sys
sys.path.append("./classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_Autoencoder_V2 import Autoencoder
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_SVM_V4 import SVM

#Save the header for the CSV file
def saveHeader(saveAs):
    #save the header
    values = [["iteration", "gamma", "nu", "tau", "lambda", "noise", "TPR", "TNR", "accuracy", "TP", "FP", "FN", "TN", "total in train", "total not in train"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 

#Save the results of testing 
def save(saveAs, iteration, gamma, nu, threshold, lambdaVal, noise, results):
    #get values:
    acc1 = results[0]
    acc2 = results[1]    
    accuracy = (results[2][2] + results[3][1]) / (results[2][0] + results[3][0])
    tp = results[2][2]
    fp = results[3][2] + results[3][3]
    fn = results[2][1] + results[2][3]
    tn = results[3][1]
    total_in_train = results[2][0]
    total_not_in_train = results[3][0]
                
    values = [[iteration, gamma, nu, threshold, lambdaVal, noise, acc1, acc2, accuracy, tp, fp, fn, tn, total_in_train, total_not_in_train]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")  

pathData = "../data"
fileData = "data_cond1_c2.csv"

#load the dataset
print("--> loading the dataset")
dataset = Dataset(pathData, fileData)

#Create noisy datasets
print("creating noisy datasets")
datasetNoise5 = Dataset(pathData, fileData)
datasetNoise5.turnToNoisy(5)            
datasetNoise10 = Dataset(pathData, fileData)
datasetNoise10.turnToNoisy(10)
print("--> datasets ready")

#CSV file
saveAs = "./summary.csv"
saveHeader(saveAs)

#parameters
numberPeopleTraining = 24
percentUnitsTraining = 0.75
learningRate = 0.001
alpha = 1.0
beta = 1.0
lambdas = [0.0, 0.1]
thresholds = [-0.15, -0.05]
gammas = [1.0, 1.25]
nus = [0.15, 0.2]
epochs =  40
batchSize = 64
iterations= 20

#Repeat training 
for i in range(1, iterations+1):
    #Split the dataset for: training, validation, unknown, test
    dataset.randomSplit(numberPeopleTraining, percentUnitsTraining)
    dataset.saveSets(".","datasets.npy")
    datasetNoise5.loadSets(".","datasets.npy")
    datasetNoise10.loadSets(".","datasets.npy")
    
    print("--> Getting the datasets for training")
    x_train, y_train, c_train = dataset.getDataset(dataset.trainingSet, batchSize=batchSize)
    x_valid, y_valid, c_valid = dataset.getDataset(dataset.validationSet, batchSize=batchSize)
    
    #Train SVM for each lambda
    for lambdaVal in lambdas:
        print("\n--> Iteration: ", i, " lambda: ", lambdaVal)
               
        print("--> training the autoencoder")
        autoencoder = Autoencoder(dataset.unitSize(), alpha, beta, lambdaVal, learningRate)
        outerModel, encoder = autoencoder.getAutoencoderCNN()
        outerModel.fit(x_train, y={"encoder": y_train, "decoder": c_train},
               batch_size = batchSize, epochs = epochs, 
               validation_data = (x_valid, {"encoder": y_valid, "decoder": c_valid}))
        
        print("--> Predicting embeddings")
        embeddings0 = EmbeddingsDataset(encoder, dataset)
        embeddings0.predictEmbeddings()            
        embeddings5 = EmbeddingsDataset(encoder, datasetNoise5)
        embeddings5.predictEmbeddings()
        embeddings10 = EmbeddingsDataset(encoder, datasetNoise10)
        embeddings10.predictEmbeddings()
                        
        #Get the random training set
        randomTrainingSet = dataset.selectRandomTrainingUnits(10)

        #Train SVM with different gammas
        for gamma in gammas:
            for nu in nus:
                print("--> Train SVM: gamma=", gamma, " nu=", nu)
                svm = SVM(embeddings0)
                svm.fit(randomTrainingSet, gamma=gamma, nu = nu)

                for tau in thresholds:                
                    print("--> Test Noise 0%, tau=", tau)
                    svm.embeddings = embeddings0                    
                    results = svm.accuracy(dataset.validationSet, dataset.unseenSet, tau)
                    save(saveAs, i, gamma, nu, tau, lambdaVal, 0.00, results)
                    
                    print("--> Test Noise 5%, tau=", tau)
                    svm.embeddings = embeddings5
                    results = svm.accuracy(dataset.validationSet, dataset.unseenSet, tau)
                    save(saveAs, i, gamma, nu, tau, lambdaVal, 0.05, results)
                        
                    print("--> Test Noise 10%,  tau=", tau)                        
                    svm.embeddings = embeddings10
                    results = svm.accuracy(dataset.validationSet, dataset.unseenSet, tau)
                    save(saveAs, i, gamma, nu, tau, lambdaVal, 0.10, results)
                        
