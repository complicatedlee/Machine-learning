#--coding:utf-8
#################################################  
# Perceptron: Perceptron
# Author : Li Xiang  
# Date   : 2018-11-08
# Github : https://github.com/complicatedlee 
# Refs: https://blog.csdn.net/wds2006sdo/article/details/51923546
#################################################  

import numpy as np 
from matplotlib import pyplot as plt 
import csv


lr = 1
max_epoch = 10  #训练多少个epochs，即全部样本训练多少次

class Perceptron(object):
    def __init__(self, x, y):
        self.train_x = x
        self.train_y = y

        #训练样本个数
        self.N = self.train_y.shape[0]

        #选去初值w, b
        self.w = np.zeros((1, 2))
        self.b = 0

    def train(self):
        c_iter = 0  #迭代次数

        while True:
            for i in range(self.N):
                tmp = self.train_y[i] * (np.dot(self.w, self.train_x[i].T) + self.b)

                #如果 yi(w*xi + b) <= 0, 更新w, b
                if tmp <= 0:
                    self.w = self.w + lr * self.train_y[i] * self.train_x[i]
                    self.b = self.b + lr * self.train_y[i]

            c_iter += 1
            print('{} epoch, w and b are updating...{}'.format(c_iter, self.w, self.b))

            if c_iter >= max_epoch:
                break
        
        return self.w, self.b
        

    def predict(self, feature):
        feature = np.array(feature)

        fea = np.dot(self.w, feature) + self.b
        if fea >= 0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    print("===============Step 1: Loading data===============")
    #Training samples
    X = np.array([[3,3],[4,3],[1,1]])
    Y = np.array([[1],[1],[-1]])

    per = Perceptron(X, Y)

    #训练样本
    print('===============Step 2: Starting training================\n')
    w, b = per.train()
    print('w, b are', w, b)

    #预测特征
    print('===============Step 3: Starting testing================')
    feature = [3, 3]
    result = per.predict(feature)   

    print(feature, 'belongs to label', result)


    #画出样本点
    for i in range(Y.shape[0]):  
        if Y[i] == -1:  
            plt.plot(X[i, 0], X[i, 1], 'or')  
        elif Y[i] == 1:  
            plt.plot(X[i, 0], X[i, 1], 'ob') 

    #设置刻度
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    #画感知机分离超平面
    x = np.arange(0, 5, 1)
    y = -(w[:, 0] / w[:, 1]) * x - b / w[:, 1]
    plt.plot(x, y)

    plt.show()   
    
