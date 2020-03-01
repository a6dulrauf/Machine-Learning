import math
import numpy as np


class LogisticRegression(object):

    def __init__(self):
        pass

    def setData(self, data):
        if data != []:
            self.data = np.array([[np.random.randint(low=0, high=2)] + x for x in data])
            self.weights = np.ones(self.data.shape[1] - 1)
            self.m = self.data.shape[0]

    def solve(self):
        pass

    # w0+w1*x1+...
    def equation(self, x):
        h = self.weights.dot(x)
        print(h)

    # 1/1+e-equation
    def sigmoid(self, equation):
        return 1 / (math.exp(-equation) + 1)

    def cal_error(self):
        pass

    # - 1/m [ sum i=1 to m y(i) * log (equation(i)) + (1-y(i)) * log(1-(equation(i)))]
    def cost(self):
        cost_list = []
        for i in range(self.m):
            cost = -1/self.m * (sum(self.data[i][0] * math.log(self.equation(self.data[:, 1:][i]))
                                 + ((1-self.data[i][0]) * math.log(1-self.equation(self.data[:,1:][i])))))
            print(cost)

if __name__ == '__main__':
    data = [[1, 3, 5],
            [2, 4, 5],
            [7, 8, 5]]
    logistic = LogisticRegression()
    logistic.setData(data)

    print(logistic.data[:, 1:])
    print(logistic.data[:, 1:][1])
    '''
    print(logistic.data[:, 0])
    print(logistic.data.shape[0])
    print(logistic.weights)
    
    print(logistic.data[0][0])
    print(logistic.data[1][0])
    print(logistic.data[2][0])
    '''
    #logistic.equation(logistic.data[:, 1:][0])
    print(logistic.cost())
