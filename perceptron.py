import numpy as np
import math

class Perceptron(object):
    def __init__(self,data=[],handlers=[],learningRate=1.0):
        self.data = np.array(data)
        self.handlers = handlers
        self.learningRate = learningRate
        self.weights = np.zeros(3)
    
    def fireHandler(self,dataPoint,finished):
        for handler in self.handlers:
            handler(self.weights,dataPoint,finished)

    def solve(self):
        self.weights = np.zeros(self.data.shape[1])
        mistakes = 1
        while(mistakes > 0):
            mistakes = 0
            for row in self.data:
                result = self.weights[0] + self.weights[1]*row[0]+self.weights[2]*row[1]
                if result * row[2] <= 0:
                    mistakes += 1
                    self.weights[1] = self.weights[1] + row[0]*row[2]
                    self.weights[2] = self.weights[2] + row[1]*row[2]
                    self.weights[0] = self.weights[0] + row[2]
                
                self.fireHandler(row,False)
        self.fireHandler([],True)

    def calculate_boundary(self,data):
        slope = -(self.weights[0]/self.weights[2])/(self.weights[0]/self.weights[1])  
        intercept = -self.weights[0]/self.weights[2]
        if math.isnan(slope):
            slope = 0.0
        if math.isnan(intercept):
            intercept = 0.0
        y =  slope * data + intercept
        return y


    def predict(self,data):
        return np.sign(self.weights[0] + self.weights[1]* data[:,0] + self.weights[2]*data[:,1])


    