import numpy as np
import math

class NaiveBayes(object):
    def __init__(self,data=[],handlers=[]):
        self.data = np.array(data)
        self.handlers = handlers
        self.classStatsMap = {}
        self.shape = self.data.shape
        self.classes = []

    def setData(self,data):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.classes = np.unique(self.data[:,self.shape[1]-1])
        self.calculateClassStatistics()

    def calculateMean(self,listOfNumbers):
        return np.mean(listOfNumbers)
    
    def calculateSD(self,listOfNumbers,mean): 
        sum = np.sum( (listOfNumbers - mean)**2 )
        return (sum/(len(listOfNumbers)-1))**.5
    
    def calculateClassStatistics(self):
        max1 = self.data.shape[1]
        for i in range(max1-1):
            self.classStatsMap[i] = {}
            for clss in self.classes:
                tmp = np.array([x[i] for x in self.data if x[max1-1] == clss])
                mean = self.calculateMean(tmp)
                std = self.calculateSD(tmp,mean)
                self.classStatsMap[i][clss] = (mean,std)
    
    def calculateDensityOfAColumnGivenValue(self, value,column,clss):
        expValue = -(value - self.classStatsMap[column][clss][0])**2/(2* self.classStatsMap[column][clss][1]**2)
        coeff = 1 / (self.classStatsMap[column][clss][1] * math.sqrt(2*math.pi))
        return coeff * math.exp(expValue)
    
    def calculateProbability(self,data):
        returnProbability = {}
        for clss in self.classes:
            prob = 1.0
            for col in range(self.shape[1]-1):
                prob = prob * self.calculateDensityOfAColumnGivenValue(data[col],col,clss)
            returnProbability[clss] = prob
        return returnProbability
    
    def predictClass(self,data):
        returnProbability = self.calculateProbability(data)
        maxProb = -1
        clss = 0
        for x in returnProbability:
            if maxProb < returnProbability[x]:
                maxProb = returnProbability[x]
                clss = x
        return clss

    def fireHandlers(self,count,total):
        for handle in self.handlers:
            handle(count,total)
    
    def predictClassVector(self,dataVector):
        returnList = []
        count = 0
        total = dataVector.shape[0]
        for data in dataVector:
            count += 1
            self.fireHandlers(count,total)
            returnProbability = self.calculateProbability(data)
            maxProb = -1
            clss = 0
            for x in returnProbability:
                if maxProb < returnProbability[x]:
                    maxProb = returnProbability[x]
                    clss = x
            returnList.append(clss)
        return returnList

if __name__ == "__main__":
    data = [[85,85,0],
    [80,90,0],
    [83,86,1],
    [70,96,1],
    [68,80,1],
    [65,70,0],
    [64,65,1],
    [72,95,0],
    [69,70,1],
    [75,80,1],
    [75,70,1],
    [72,90,1],
    [81,75,1],
    [71,91,0]
    ]
    nb = NaiveBayes(data,[])
    nb.calculateClassStatistics()
    print(nb.classStatsMap)
    print(nb.calculateDensityOfAColumnGivenValue(62,1,0))
    print(nb.calculateProbability([60,62]))
    print(nb.predictClass([60,62]))
