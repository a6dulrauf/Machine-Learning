import numpy as np
import math


class LogisticRegression(object):
    def __init__(self, data=[], handlers=[], learningRate=.01, iterations=2000, threshold=0.000005):
        self.data = np.array([[1] + x for x in data])
        self.handlers = handlers
        self.learningRate = learningRate
        if data == []:
            self.weights = np.zeros(3)
        else:
            self.weights = np.zeros(self.data.shape[1])
        self.iterations = iterations
        self.threshold = threshold

    def setData(self, data):
        if data != []:
            self.data = np.array([[1] + x for x in data])
            self.weights = np.zeros(self.data.shape[1] - 1)

    def fireHandler(self, finished):
        for handler in self.handlers:
            handler(self.weights, finished)

    def sigmoid(self, sum):
        return (math.exp(sum)) / (math.exp(sum) + 1)

    def solve(self):

        iterationCount = 0
        outputCol = self.data.shape[1] - 1
        while (iterationCount < self.iterations):
            iterationCount += 1

            tempWeights = np.copy(self.weights)

            for col in range(self.data.shape[1] - 1):
                sum = 0
                for row in self.data:
                    sum += (row[outputCol] - self.sigmoid(self.weights.dot(row[:-1]))) * row[col]
                tempWeights[col] = self.weights[col] + self.learningRate * sum

            self.threshold = abs(np.sum(tempWeights) - np.sum(self.weights))

            self.weights = np.copy(tempWeights)

            self.fireHandler(False)
        self.fireHandler(True)

    def calculate_boundary(self, data):

        slope = -(self.weights[0] / self.weights[2]) / (self.weights[0] / self.weights[1])
        intercept = -self.weights[0] / self.weights[2]
        if math.isnan(slope):
            slope = 0.0
        if math.isnan(intercept):
            intercept = 0.0
        y = slope * data + intercept
        # print("slope = %f, intercept = %f"%(slope,intercept))
        return y

    def predict(self, data):
        result = []
        for row in data:
            nt = [1] + row.tolist()

            if self.sigmoid(self.weights.dot(np.array(nt))) >= 0.5:
                result.append(1)
            else:
                result.append(-1)
        return result


if __name__ == "__main__":
    data = [[1, 3, 5],
            [2, 4, 5],
            [7, 8, 5]]
    lg = LogisticRegression()
    lg.setData(data)
    # lg.solve()
    print(lg.data.shape)
    l = [1, 2, 3, 4, 5]
    print(lg.data)
    print(data)
