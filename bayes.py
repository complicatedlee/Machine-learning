#--coding:utf-8
#################################################  
# NaiveBayes: NaiveBayes
# Author : Li Xiang  
# Date   : 2018-07-28 
# Github : https://github.com/complicatedlee 
# Refs: https://www.cnblogs.com/yiyezhouming/p/7364688.html
#       https://www.zybuluo.com/zsh-o/note/1205353
#################################################  
import numpy as np

class NaiveBayes(object):
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.Y = train_y
        self.N, self.n = self.X.shape #N样本个数, n特征属性个数
        self.label = np.max(self.Y) + 1 #样本标签个数（0,1,...）
        self.P_y = np.zeros(self.label) #先验概率
    
    def classify(self, features):
        S_j = np.max(self.X[:, 0]) + 1 #第j个特征可能取的值，即第i个特征属性中的不重复值的个数

        #Step 1: 求先验概率
        for i in range(self.label):
            self.P_y[i] = len(np.where(Y[:, 0] == i)[0]) / self.N
        
        P_x_y = []
        #Step 2: 求条件概率
        for i in range(self.n):
            x_y = np.zeros((self.label, S_j))
            for j in range(self.label):
                t = X[np.where(Y[:, 0] == j)[0]] #训练数据X中标签等于j的数据
                for l in range(S_j):
                    x_y[j, l] = len(np.where(t[:, i] == l)[0]) / len(np.where(Y[:, 0] == i)[0]) #训练数据中标签等于j的，又等于特征的第l个值得数据
            P_x_y.append(x_y)
        
        #Step 3: 计算给定的实例
        Py_x = np.zeros(self.label)
        for i in range(self.label):
            Py_x[i] = 1.
            for j in range(self.n):
                Py_x[i] = Py_x[i] * P_x_y[j][i, features[j]]
            Py_x[i] = Py_x[i] * self.P_y[i]
        
        #Step 4: 确定实例的类别
        prob_label = np.argmax(Py_x)
        return prob_label
        

if __name__ == '__main__':
    #Training samples
    X = np.array([[0,0],[0,1],[0,1],[0,0],[0,0],[1,0],[1,1],[1,1],[1,2],[1,2],[2,2],[2,1],[2,1],[2,2],[2,2]])
    Y = np.array([[0],[0],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[0]])

    nb = NaiveBayes(X, Y)
    # x1,x2
    features = [1,0]
    # 该特征应属于哪一类
    result = nb.classify(features)
    print(features,'belongs to label',result)