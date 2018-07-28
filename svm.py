#--coding:utf-8
#################################################  
# self: Linear and RBF support vector machine  
# Author : Li Xiang  
# Date   : 2018-07-25 
# Github : https://github.com/complicatedlee 
# Refs: https://blog.csdn.net/m0_37269455/article/details/74201689
#       https://www.zybuluo.com/zsh-o/note/1219660
#################################################  

import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

epsilon = 1e-8

class self(object):
    """This implements linear self and RBF-self.
    Parameters
    ----------
    kernel_opt : string, optional (default='rbf'). Specifies the kernel type to be used in the algorithm.
    C : soft margin parameter    
    tol : stop threshold
    maxIter : maxIter
    """
    
    def __init__(self, kernel_opt = 'rbf', C = 1., toler = 1e-3, maxIter = 100):
        self.kernel_opt = kernel_opt
        self.C = C
        self.tol = toler
        self.b = 0.
        self.maxIter = maxIter


    def kernelValue(self, matrix_x, sample_x, sigma = 1.):
        """Calculate kernel value.
        Parameters
        ----------
        matrix_x : training samples matrix.
        sample_x : i th traning samples vectors.
        sigma : set sigma if opt == 'rbf' (default = 1.).

        Returns
        -------
        kernelValue : different kernel opt has different kernel value.
        """
        numSamples = matrix_x.shape[0] 
        kernelValue = np.zeros((numSamples, 1))

        if self.kernel_opt == 'linear': #线性函数
            kernelValue = matrix_x * sample_x.T

        elif self.kernel_opt == 'rbf': #高斯径向基函数 radial basis function
            for i in range(numSamples):
                diff = matrix_x[i, :] - sample_x
                kernelValue[i] = np.exp((np.linalg.norm(diff)**2) / (-2.0*sigma**2)) 
        else:
            raise NameError("It must be one of 'linear', 'rbf'.")
        return kernelValue


    def separation_hyperplane_func(self, test_x):
        """Separation hyerplane w*x + b* = 0, w* = sum(alpha(i) * label(i) * x(i)), b* = label(j) - sum(alpha(i) * label(i) * (x(i) * x(j)))

        Parameters
        ----------
        test_x : test samples matrix.

        Returns
        -------
        w*x + b*
        """
        numTestSamples = test_x.shape[0]
        separa_hyper_func = np.zeros((numTestSamples, 1))
        for i in range(numTestSamples): #<统计学习方法> P133
            kernel_fuc = self.kernelValue(self.X, test_x[i,:])
            separa_hyper_func[i] = np.dot(kernel_fuc.T, np.multiply(self.alpha, self.Y)) + self.b 
        return separa_hyper_func
    
    def calcError(self, i):
        #统计学习方法 P127 （7.105）
        return float(np.multiply(self.alpha, self.Y).T * self.K[:, i] + self.b) - float(self.Y[i]) 

    # update the error cache for alpha k after optimize alpha k  
    def updateError(self, k):  
        error = self.calcError(k)  
        self.errorCache[k] = [1, error]  

    # select alpha j which has the biggest step  
    def selectAlpha_j(self, alpha_i, error_i):  
        self.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)  
        candidateAlphaList = np.nonzero(self.errorCache[:, 0])[0] # mat.A return array  
        maxStep = 0; alpha_j = 0; error_j = 0  
    
        # find the alpha with max iterative step  
        if len(candidateAlphaList) > 1:  
            for alpha_k in candidateAlphaList:  
                if alpha_k == alpha_i:   
                    continue  
                error_k = self.calcError(alpha_k)  
                if abs(error_k - error_i) > maxStep:  
                    maxStep = abs(error_k - error_i)  
                    alpha_j = alpha_k  
                    error_j = error_k  
        # if came in this loop first time, we select alpha j randomly  
        else:             
            alpha_j = alpha_i  
            while alpha_j == alpha_i:  
                alpha_j = int(np.random.uniform(0, self.N))  
            error_j = self.calcError(alpha_j)  
        
        return alpha_j, error_j  


    def optimize_a(self, alpha_i):
        """For optimizing alpha i and j
        """
        error_i = self.calcError(alpha_i)
        ### check and pick up the alpha who violates the KKT condition  
        ## satisfy KKT condition  
        # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
        # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
        # 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
        ## violate KKT condition  
        # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
        # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
        # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
        # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
        if (self.Y[alpha_i] * error_i < -self.tol) and (self.alpha[alpha_i] < self.C) or (self.Y[alpha_i] * error_i > self.tol) and (self.alpha[alpha_i] > 0):  
    
            # step 1: select alpha j  
            alpha_j, error_j = self.selectAlpha_j(alpha_i, error_i)   #随机选取aj，并返回其E值
            alpha_i_old = self.alpha[alpha_i].copy()  
            alpha_j_old = self.alpha[alpha_j].copy()  
    
            # step 2: calculate the boundary L and H for alpha j 
            #以下代码的公式参考《统计学习方法》p126 
            if self.Y[alpha_i] != self.Y[alpha_j]:  
                L = max(0, self.alpha[alpha_j] - self.alpha[alpha_i])  
                H = min(self.C, self.C + self.alpha[alpha_j] - self.alpha[alpha_i])  
            else:  
                L = max(0, self.alpha[alpha_j] + self.alpha[alpha_i] - self.C)  
                H = min(self.C, self.alpha[alpha_j] + self.alpha[alpha_i])  
            if L == H:  
                return 0  
    
            # step 3: calculate eta (the similarity of sample i and j) 
            # 公式参考《统计学习方法》p127 (7.107) 
            eta = 2.0 * self.K[alpha_i, alpha_j] - self.K[alpha_i, alpha_i] - self.K[alpha_j, alpha_j]  
            if eta >= 0:  
                return 0  
    
            # step 4: update alpha j  
            # 公式参考《统计学习方法》p127 (7.106) 
            self.alpha[alpha_j] -= self.Y[alpha_j] * (error_i - error_j) / eta  
    
            # step 5: clip alpha j  
            # 公式参考《统计学习方法》p127 (7.108) 
            if self.alpha[alpha_j] > H:  
                self.alpha[alpha_j] = H  
            if self.alpha[alpha_j] < L:  
                self.alpha[alpha_j] = L  
    
            # step 6: if alpha j not moving enough, just return   ???     
            if abs(alpha_j_old - self.alpha[alpha_j]) < 0.00001:  
                self.updateError(alpha_j)  
                return 0  
    
            # step 7: update alpha i after optimizing aipha j  
            # 公式参考《统计学习方法》p127 (7.109) 
            self.alpha[alpha_i] += self.Y[alpha_i] * self.Y[alpha_j] * (alpha_j_old - self.alpha[alpha_j])  
    
            # step 8: update threshold b  
            # 公式参考《统计学习方法》p129 (7.114-7.116) 
            b1 = self.b - error_i - self.Y[alpha_i] * (self.alpha[alpha_i] - alpha_i_old) * self.K[alpha_i, alpha_i] - self.Y[alpha_j] * (self.alpha[alpha_j] - alpha_j_old) * self.K[alpha_i, alpha_j]  
            b2 = self.b - error_j - self.Y[alpha_i] * (self.alpha[alpha_i] - alpha_i_old) * self.K[alpha_i, alpha_j] - self.Y[alpha_j] * (self.alpha[alpha_j] - alpha_j_old) * self.K[alpha_j, alpha_j]  
            if (0 < self.alpha[alpha_i]) and (self.alpha[alpha_i] < self.C):  
                self.b = b1  
            elif (0 < self.alpha[alpha_j]) and (self.alpha[alpha_j] < self.C):  
                self.b = b2  
            else:  
                self.b = (b1 + b2) / 2.0  
    
            # step 9: update error cache for alpha i, j after optimize alpha i, j and b  
            self.updateError(alpha_j)  
            self.updateError(alpha_i)  
    
            return 1  
        else:  
            return 0  
    

    
    def fit(self, X, Y):
        """Fit the self model according to the given training data.

        Parameters
        ----------
        X : {array-like, matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Y : array-like, shape = [n_samples]
            Target values (integers in classification)
        """
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]  #训练样本数
        self.alpha = np.zeros((self.N, 1)) #对偶问题的最优解
        self.K = np.mat(np.zeros((self.N, self.N)))
        for i in range(self.N):
            self.K[:, i] = self.kernelValue(X, X[i, :])
        
        
        self.errorCache = np.zeros((self.N, 2))
        
        # start training (SMO)
        entireSet = True
        alphaPairsChanged = 0
        iterCount = 0
        # Iteration termination condition:  
        #   Condition 1: reach max iteration  
        #   Condition 2: no alpha changed after going through all samples,  
        #                in other words, all alpha (samples) fit KKT condition  
        while (iterCount < self.maxIter) and (alphaPairsChanged > 0) or entireSet:
            alphaPairsChanged = 0

            #update alpha over all training examples
            if entireSet:
                for i in range(self.N):
                    alphaPairsChanged += self.optimize_a(i)
                print('Iter:{} entire set, alpha pairs changed:{}'.format(iterCount, alphaPairsChanged))
                iterCount += 1
            
            # update alpha over examples where alpha is not 0 & not C (not on boundary)  
            # 所谓非边界变量，就是指满足 0<αi<C 的变量
            # 因为随着多次子优化过程，边界变量倾向于留在边界，而非边界变量倾向于波动，
            # 这一步启发式的选择算法是基于节省时间考虑的，并且算法会一直在非边界变量集合上遍历，
            # 直到所有非边界变量都满足KKT条件（self-consistent） 
            else:  
                for j in np.where((self.alpha != 0) & (self.alpha != self.C))[0]: 
                    alphaPairsChanged += self.optimize_a(i)  
                print('---iter:{} non boundary, alpha pairs changed:{}'.format(iterCount, alphaPairsChanged))  
                iterCount += 1  
    
            # alternate loop over all examples and non-boundary examples  
            if entireSet:  
                entireSet = False  
            elif alphaPairsChanged == 0:  
                entireSet = True  


    def classification_decision_func(self, test_x):
        return np.sign(self.separation_hyperplane_func(test_x))

    # testing svm model 
    def testSVM(self, test_x, test_y):  
        numTestSamples = test_x.shape[0]  
        supportVectorsIndex = np.nonzero(self.alpha > 0)[0]  
        supportVectors      = self.X[supportVectorsIndex]  
        supportVectorLabels = self.Y[supportVectorsIndex]  
        supportVectorAlphas = self.alpha[supportVectorsIndex]  
        matchCount = 0  
        for i in xrange(numTestSamples):  
            kernelValue = self.kernelValue(supportVectors, test_x[i, :])  
            predict =  np.dot(kernelValue.T, np.multiply(supportVectorLabels, supportVectorAlphas)) + self.b  
            if np.sign(predict) == np.sign(test_y[i]):  
                matchCount += 1  
        accuracy = float(matchCount) / numTestSamples  
        return accuracy  
        

    # show your trained svm model only available with 2-D data  
    def showSVM(self, resolution=100, colors=('b', 'k', 'r')):  
        if self.X.shape[1] != 2:  
            print "Sorry! I can not draw because the dimension of your data is not 2!"  
            return 1  
    
        # draw all samples  
        for i in range(self.N):  
            if self.Y[i] == -1:  
                plt.plot(self.X[i, 0], self.X[i, 1], 'or')  
            elif self.Y[i] == 1:  
                plt.plot(self.X[i, 0], self.X[i, 1], 'ob')  
    
        # mark support vectors  
        supportVectorsIndex = np.nonzero(self.alpha > 0)[0]  
        for i in supportVectorsIndex:  
            plt.plot(self.X[i, 0], self.X[i, 1], 'oy')  
        
        # draw the classify line  
        x_range = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
        y_range = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
        grid = [[self.separation_hyperplane_func(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
        grid = np.array(grid).reshape(len(x_range), len(y_range))
        
        plt.contour(x_range, y_range, grid.T, (0,), linewidths=(1,),
                linestyles=('-',), colors=colors[1])
        plt.show()  



if __name__ == "__main__":
    
    print("Step 1: Loading data...")
    #X: training samples
    #Y: labels (0 or 1)
    X, Y = make_moons(n_samples=100, shuffle=True, noise=0.1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y[Y == 0] = -1

    train_x = X[:80, :]
    train_y = Y[:80, np.newaxis]
    test_x = X[80:101, :]
    test_y = Y[80:101, np.newaxis]

    print("Step 2: Training...")
    svm = self(C = 1)
    svm.fit(train_x, train_y)

    print("Step 3: Testing...")
    accuracy = svm.testSVM(test_x, test_y)

    print("Step 4: Showing result...")
    print("The classify accuracy is:{}".format(accuracy * 100))
    svm.showSVM()

