# Project: Prototyping-Encoder-Decoder-with-Triplet-Loss
# version: 1.0
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
# Date: June 2020
#
#Include a reference to this site if you will use this code.

import numpy as np
from sklearn import svm
from sklearn.svm import OneClassSVM
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset

class SVM:
    #constructor
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.classifier = {}
        self.centroids = {}
        self.results = None
        
    #trains instances of SVM.svc according to dataIndex information
    #Turn in to a binary classification problem, label 1 for current_id, and -1 for the rest    
    def fit(self, dataIndex, gamma=0.1, C=1.0, classWeight=True, nu = 0.5):
        self.results=None
        self.classifier.clear()
        self.centroids = self.embeddings.getCentroids(dataIndex)
        x, y = self.embeddings.getDataset(dataIndex)
        for key in dataIndex:
            yBinary = [1 if userId==key else -1 for userId in y]
            yBinary = np.asarray(yBinary)
            x_current = x[yBinary == 1]           
            svc = OneClassSVM(kernel='rbf', gamma=gamma, nu = nu)
            self.classifier[key] = svc.fit(x_current)
        return(self.classifier, self.centroids)        
    
    #returns the accuracy of the trained network
    def accuracy(self, dataIndexSeen, dataIndexUnseen, threshold=-1):
        score1 = self.test(dataIndexSeen, threshold)
        score2 = self.test(dataIndexUnseen, threshold)  
        acc1 = score1[2] / score1[0]
        acc2 = score2[1] / score2[0]
        self.results = [acc1, acc2, score1, score2]
        return(self.results)
            
    #Test the results of SVM.svc
    def test(self, dataIndexTest, threshold):
        score = [0,0,0,0]
        for key in dataIndexTest:
            for unitId in dataIndexTest[key]:
                embeddedVector = self.embeddings.getEmbedded(key, unitId)
                
                closest = self.getClosestCentroid(embeddedVector) 
                if threshold == None:
                    predicted = self.classifier[closest].predict([embeddedVector]) 
                    if predicted == -1: closest = -1
                else:
                    predicted = self.classifier[closest].decision_function([embeddedVector])
                    
                    if predicted[0] * threshold > 0 and predicted[0] > threshold: 
                        closest = key
                    else:
                        if predicted[0] > 0: closest = key
                        if predicted[0] < 0: closest = -1
                    
                case=1 if closest==-1 else 2 if closest==key else 3
                score[case] +=1  #counter of each case
                score[0]+=1      #total of tests
        return score
        
    ##find the closest centroid. returns the user id who has the closest centroid
    def getClosestCentroid(self, embeddedVector):
        closest = -1
        minDist = float("inf")
        for key in self.centroids:
            dist = np.linalg.norm(embeddedVector - self.centroids[key]) #max ~1.9
            if dist < minDist:
                minDist = dist
                closest = key
        return(closest)
    
    #Prints the accuracy according to the last obtained results
    def printAccuracy(self):
        print("Accuracy validation dataset=", self.results[0])
        print(" - Total in train=", self.results[2][0])
        print(" - in train incorrect classify=", self.results[2][1])
        print(" - in train correct identify=", self.results[2][2])
        print(" - in train incorrect identify=", self.results[2][3])

        print("Accuracy unseen dataset =", self.results[1])
        print(" - total not in train=", self.results[3][0])
        print(" - not in train correct classify=", self.results[3][1])
        print(" - not in train incorrect classify=", self.results[3][2]+self.results[3][3])
            
    #Save the accuracy as a nampy array
    def saveAccuracy(self, path, fileName):
        np.save(path+"/"+fileName, self.results)
        
