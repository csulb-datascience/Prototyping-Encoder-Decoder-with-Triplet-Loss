import numpy as np
import random
import matplotlib.pyplot as plt

from itertools import combinations
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset

class TripletPairs:

    #Constructor
    def __init__(self, embeddings, dataIndex):
        self.embeddings = embeddings
        self.dataIndex = dataIndex
        
    #Creates the Anchor, Positive pairs
    def getPairsAP(self):
        result = dict()
        for key in self.dataIndex:
            current_set = self.dataIndex[key]
            if len(current_set) != 0:
                #get all combinations picking 2
                comb_index = list(combinations(current_set, 2))
                record = [item + (key,) for item in comb_index]
                result[key] = record
        return result

    #Creates the Anchor, Negative pairs
    def getPairsAN(self, A_P_pairs):
        result = dict()
        for key in (self.dataIndex.keys() & A_P_pairs.keys()):
            pairs = set()
            while len(pairs) < len(A_P_pairs[key]):
                #randomly pick anchor unit step                
                anchor = random.choice(self.dataIndex[key])
                for negative_key in (A_P_pairs.keys() - {key}):
                    #Pick a random negative unit step in the list                            
                    negative = random.choice(self.dataIndex[negative_key])
                    pairs.add((anchor, negative, key, negative_key))
                    if len(pairs) >= len(A_P_pairs[key]): break                        
            result[key] = list(pairs)
        return result
                        
    #Returns the pairs encoded
    def encodePairs(self, pairsData):
        result = []
        for key in pairsData:
            for item in pairsData[key]:
                idxPair = 2 if len(item) == 3 else 3
                encodedAnchor = self.embeddings.getEmbedded(item[2], item[0])
                encodedPair = self.embeddings.getEmbedded(item[idxPair], item[1])
                result.append((encodedAnchor, encodedPair))
        return result

    #generates the encoded pairs from the index dataset
    def getEncodedPairs(self):
        A_P_pairs = self.getPairsAP()
        A_N_pairs = self.getPairsAN(A_P_pairs)
        A_P_encoded = self.encodePairs(A_P_pairs)
        A_N_encoded = self.encodePairs(A_N_pairs)
        return(A_P_encoded, A_N_encoded)
    
    #Given a list of pair of images, either (A, P) or (A, N) (Encoded) only
    #return the list of distances between two unit steps
    def calculateDistances(self, pairsEncoded):
        return [np.linalg.norm(item[0] - item[1]) for item in pairsEncoded]

    #return the list of distances between two unit steps from a index dataset
    def getPairsDistance(self):
        encodedAP, encodedAN = self.getEncodedPairs()
        distanceAP = self.calculateDistances(encodedAP) # a list of positive distances(A, P)
        distanceAN = self.calculateDistances(encodedAN) # a list of negative distance(A, N)
        return(distanceAP, distanceAN)
    
    #plot the histogram
    def plotHistogram(self, distances=None, label=" ", typeStep=True): 
        if distances==None: distances = self.getPairsDistance()
        bins = np.linspace(0, 2, 1000)
        labels = [label+'_P', label+'_N']
        if typeStep == True:
            plt.hist(distances, bins, label=labels, color=["blue", "orange"], density=True, histtype="step")
        else:
            plt.hist(distances, bins, label=labels, color=["blue", "orange"])

        plt.legend(loc='upper right')
        plt.xlabel(label)
        plt.show()    
        