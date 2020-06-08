import pandas as pd
import numpy as np
import random

class Dataset:
    def __init__(self, path, fileName):
        self.user = []
        self.unitsMap = {}
        self.trainingSet = {}
        self.validationSet ={}
        self.unseenSet={}
        self.testSet ={}
        self.loadData(path, fileName)

    #return the size of the unit step if there is at least a user in the dataset
    def unitSize(self):
        size = 0 if len(self.user)==0 else len(self.unitsMap[self.user[0]][0])
        return(size)
    
    #return the indicated unit step 
    def getUnitStep(self, personId, unitId):
        if personId in self.unitsMap:
            if unitId < len(self.unitsMap[personId]):
                return(self.unitsMap[personId][unitId])
        return(None)

    #Save the datasets on disk
    def saveSets(self, path, fileName):
        summary = {"training":self.trainingSet,"validation":self.validationSet,
                   "unseen":self.unseenSet, "test":self.testSet}
        np.save(path + "/" + fileName, summary)
        
    #load the dataset index saved with saveSets
    def loadSets(self, path, fileName):
        summary = np.load(path + "/" + fileName, allow_pickle='TRUE').item()
        self.trainingSet = summary["training"]
        self.validationSet = summary["validation"]
        self.unseenSet = summary["unseen"]
        self.testSet = summary["test"]
                 
    #Reads the dataset from disk and creates it in memory as a dictionary
    def loadData (self, path, fileName):
        data = pd.read_csv(path + "/" + fileName)
        self.initSets()
        self.user = np.array(data['user_id'].unique()).astype(int)
        self.unitsMap = self.get_data_dictionary(data, self.user)
        
    #clean the variables related to the dataset
    def initSets(self):
        self.unitsMap.clear()
        self.trainingSet.clear()
        self.validationSet.clear()
        self.unseenSet.clear()
        self.testSet.clear()
    
    #Given a data set, and a list of ids, return a dictionary 
    #that map user id to their list of images
    def get_data_dictionary(self, data, ids):
        table = {}
        for person_id in ids:
            list_of_images = []
            current_person = data[data['user_id'] == person_id]
            unique_unit_ids = current_person['unit_id'].unique()
            for unit_id in unique_unit_ids:
                current_unit = current_person[current_person['unit_id'] == unit_id]
                current_image = current_unit.iloc[:, 4:]
                current_image = current_image.values
                list_of_images.append(current_image)
            table[person_id] = np.asarray(list_of_images)
        return table

    #splits the dataset by selecting randomly the people for training and the percentage
    #training set -> % units of numberPeopleTraining;  (% = percentTraining)
    #validation set -> (1-percentTraining) units of numberPeopleTraining
    #test set -> (1-percentTraining) units of numberPeopleTraining + 100% of remaining users            
    def randomSplit(self, numberPeopleTraining, percentUnits):
        trainingPeople = set(random.sample(list(self.user), numberPeopleTraining))
        self.trainingSet = self.getDataIndex(trainingPeople, percentUnits)
        self.validationSet = self.getDataIndex(trainingPeople, percentUnits, complement=True)
        self.unseenSet = self.getDataIndex(self.notTrainingPeople(), 1.0) #100%
        self.testSet = self.validationSet.copy()
        self.testSet.update(self.unseenSet)  #test set includes validation and unseen data
                    
    #return the set of people included in the training dataset        
    def trainingPeople(self):
        return(set(self.trainingSet.keys()))
               
    #return the set of people that is not included in the training dataset
    def notTrainingPeople(self):
        return(set(self.user) - self.trainingPeople()) 

    #return a percentage of indexes to the units of users, according to complement:
    #Complement=False returns [0..percentage] indexes from the list of units
    #Complement=True returns [percentage..length] indexes from the list of units
    def getDataIndex (self, users, percentage, complement=False):
        dataIndex = {}
        for person in users:
            limit = int(percentage * len(self.unitsMap[person]))
            endIndex = limit if not complement else len(self.unitsMap[person])
            iniIndex = 0 if not complement else limit
            dataIndex[person] = list(range(iniIndex, endIndex))
        return dataIndex                
    
    #return the total number of units in the dataset index and 
    #the number of units per Id
    def dataIndexSize(self, dataIndex):
        byId = {}
        size = 0
        for key in dataIndex:
            byId[key] = len(dataIndex[key])
            size = size + len(dataIndex[key])
        return(size, byId)
    
    #return the mean unit for each person Id from a dataset index
    def getMeanUnit(self, dataIndex, normalized=True):
        results={}
        for key in dataIndex:
            units = [self.unitsMap[key][i] for i in dataIndex[key]]
            mean = np.mean(units, axis=0)
            if normalized:
                n = np.linalg.norm(mean, axis=0)
                mean = np.divide(mean, n, out=np.zeros_like(mean), where=n!=0)                
            results[key] = mean
        return(results)
    
    #returns a random binary matrix with a percentage of zeroes
    def getNoiseMatrix(self, shape2D, noisePercent):
        N= shape2D[0] * shape2D[1]
        K = (N*noisePercent)//100
        arr = np.array([0] * K + [1] * (N-K))
        np.random.shuffle(arr)
        return(arr.reshape(shape2D))        
        
    #Transform the dataset to a noisy dataset
    def turnToNoisy(self, noisePercent=0):
        for key in self.unitsMap:
            for i in range(len(self.unitsMap[key])):
                unit = np.array(self.unitsMap[key][i])
                noise = self.getNoiseMatrix(unit.shape, noisePercent)
                self.unitsMap[key][i] = unit * noise
        
    #Convert a index dataset to array dataset        
    def toArray(self, dataIndex, noisePercent=0, normalized=True):
        unitMean = self.getMeanUnit(dataIndex, normalized)
        x, y, m = [],[],[]
        for key in dataIndex.keys():
            for i in dataIndex[key]:
                unit = np.array(self.unitsMap[key][i])
                noise = self.getNoiseMatrix(unit.shape, noisePercent)                
                y.append(key)
                x.append(unit * noise)
                m.append(unitMean[key])
        return(np.asarray(x), np.asarray(y), np.asarray(m))
    
    # Returns the dataset where the units are interleaved: 
    # one unit person 1, one unit person 2, ..., one unit person n, one unit person 1, ...
    def toArrayInterleaved(self, dataIndex, noisePercent=0, normalized=True):
        unitMean = self.getMeanUnit(dataIndex, normalized)
        x, y, m = [], [], []
        counter, numUnits = 0, 0
        totalUnits, _ = self.dataIndexSize(dataIndex)        
        while (numUnits < totalUnits):
            for key in dataIndex.keys():
                if counter < len(dataIndex[key]):
                    unit = np.array(self.unitsMap[key][counter])
                    noise = self.getNoiseMatrix(unit.shape, noisePercent)                                    
                    x.append(unit * noise)
                    y.append(key)
                    m.append(unitMean[key])
                    numUnits +=1                    
            counter += 1 
        return(np.asarray(x), np.asarray(y), np.asarray(m))

    #return the dataset generated by an index dataset
    def getDataset(self, dataIndex, noisePercent=0, interleaved=True):
        x, y, m = self.toArrayInterleaved(dataIndex, noisePercent) \
               if interleaved else self.toArray(dataIndex, noisePercent) 
        press_x = np.asarray(x[:, :, 0:16])
        acc_x = np.asarray(x[:, :, 16:22])
        gyro_x = np.asarray(x[:, :, 22:])
        return([press_x, acc_x, gyro_x], y)

    #return the dataset with an additional Y added to X
    def getDatasetAugmented(self, dataIndex, noisePercent=0, interleaved=True, multiple=1):
        x, y, m = self.toArrayInterleaved(dataIndex, noisePercent) \
               if interleaved else self.toArray(dataIndex, noisePercent) 
        
        length = multiple * (len(y) // multiple)
        press_x = np.asarray(x[:length, :, 0:16])
        acc_x = np.asarray(x[:length, :, 16:22])
        gyro_x = np.asarray(x[:length, :, 22:])
        return([press_x, acc_x, gyro_x, y[:length], m[:length]], y[:length])
    
    #return the training dataset
    def getTrainingDataset(self, noisePercent=0):
        return(self.getDataset(self.trainingSet, noisePercent))
                
    #return the validation dataset
    def getValidationDataset(self, noisePercent=0):
        return(self.getDataset(self.validationSet, noisePercent))
    
    #Return k random units step indexes for each id
    def selectRandomUnits(self, dataIndex, k):
        result = dict()
        for key in dataIndex:
            if k >= len(dataIndex[key]):
                result[key] = dataIndex[key].copy()
            else:
                result[key] = random.sample(dataIndex[key], k)
        return result
    
    #return the training people with k random unit step indexes
    def selectRandomTrainingUnits (self, k):
        return(self.selectRandomUnits(self.trainingSet, k))
    