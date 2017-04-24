# -*- coding: utf-8 -*-

'''
This file will only work on a mac with Anaconda for Python 3.6 installed.
To do so please uncomment line 13 and computeLagrangeMultipliers on starting line 75

'''
''' Included libraries'''
import math
import sys
import os
import numpy as np
#import cvxopt
from sklearn import datasets # Used as a test in main below
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, cross_val_predict, ShuffleSplit, train_test_split


''' My code '''
from participant import Participant
from kernels import *
from principal_component_analysis import PCA

''' OS stuff '''
dirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dirPath + "/../utils")
from data_manipulation import normalize
from data_operation import accuracy_score, calculate_covariance_matrix


redPatch = mpatches.Patch(color='red', label='Hinge movement')
greenPatch = mpatches.Patch(color='green', label='Pendulum movement')
graphLegend = [greenPatch, redPatch]

# Hide cvxopt output
#cvxopt.solvers.options['show_progress'] = False
'''
Example output when 'show_progress' is true
     pcost       dcost       gap    pres   dres
 0: -1.8847e+01 -1.3222e+02  5e+02  2e+00  1e-14
 1: -1.3308e+01 -7.7172e+01  6e+01  2e-15  1e-14
 2: -1.7741e+01 -2.6323e+01  9e+00  1e-16  1e-14
 3: -2.1347e+01 -2.2921e+01  2e+00  3e-16  1e-14
 4: -2.2026e+01 -2.2231e+01  2e-01  1e-15  1e-14
 5: -2.2116e+01 -2.2131e+01  2e-02  2e-15  1e-14
 6: -2.2122e+01 -2.2123e+01  1e-03  3e-16  1e-14
 7: -2.2123e+01 -2.2123e+01  1e-04  2e-16  1e-14
 8: -2.2123e+01 -2.2123e+01  1e-06  2e-16  1e-14
'''

class SupportVectorMachine(object):
    def __init__(self, kernel, C = 0, power = 4, gamma = None, coef = 4):
        '''
        Comment all the attributes
        Kernel is the 
        C is the hyperparameter that controls the effect of the soft margin. I assume for now that 0 means no soft margin.
        '''
        self.C = C 
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagrMultipliers = None
        self.supportVectors = None
        self.supportVectorTargets = None
        self.intercept = None
        
        
        print('SVM reporting for duty!')
        print('')
        pass
    
    # Can only be used on Mac
#    def computeLagrangeMultipliers(self, X, y, kernelMatrix, nSamples):
#        # Define the quadratic optimization problem
#        # Honestly, this function is gibberish 
#        P = cvxopt.matrix(np.outer(y, y) * kernelMatrix, tc='d')
#        q = cvxopt.matrix(np.ones(nSamples) * -1)
#        A = cvxopt.matrix(y, (1, nSamples), tc='d')
#        b = cvxopt.matrix(0, tc='d')
#        
#        if not self.C:
#            G = cvxopt.matrix(np.identity(nSamples) * -1)
#            h = cvxopt.matrix(np.zeros(nSamples))
#        else:
#            G_max = np.identity(nSamples) * -1
#            G_min = np.identity(nSamples)
#            G = cvxopt.matrix(np.vstack((G_max, G_min)))
#            h_max = cvxopt.matrix(np.zeros(nSamples))
#            h_min = cvxopt.matrix(np.ones(nSamples) * self.C)
#            h = cvxopt.matrix(np.vstack((h_max, h_min)))
#        
#        # Solve the quadratic optimization problem using cvxopt
#        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
#        
#        # Lagrange multipliers
#        return np.ravel(minimization['x'])
     
    def createKernelMatrix(self, X, nSamples):
         # Calculate a matrix of values that came through the kernel
        # Fill it with zeroes first
        matrixOfKernelValues = np.zeros((nSamples, nSamples))
        # Replace each cell with it's values passed through the kernel
        for i in range(nSamples):
            for j in range(nSamples):
                # Set the matrix value to be the X data passed through the chosen kernel operation
                matrixOfKernelValues[i, j] = self.kernel(X[i], X[j])
                
    # Train the classifier            
    def fit(self, X, y):
        
        nSamples, nFeatures = np.shape(X) 
        
        # The gamma will 1/nFeatures by default, unless something else is supplied
        if not self.gamma:
            self.gamma = 1 / nFeatures
        
        # Initialize the chosen kernel method with the arguments provided
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)
        
        matrixOfKernelValues = self.createKernelMatrix(X, nSamples)
        
        lagrMult = self.computeLagrangeMultipliers(X, y, matrixOfKernelValues, nSamples)
        
        # Extract support vectors
        print('Lagrange Multipliers {}\n'.format(lagrMult))
        # Get indexes of non-zero lagr. multipiers
        lagrangeIndexes = lagrMult > 0.000001 # I read that this is better than > 0
        # Get the corresponding lagr. multipliers
        self.lagrMultipliers = lagrMult[lagrangeIndexes]
        print('Multipliers > 0 {}\n'.format(self.lagrMultipliers))
        # Find the samples that correspond as support vectors
        self.supportVectors = X[lagrangeIndexes]
        print('I have {} support vectors {}\n'.format(len(self.supportVectors),self.supportVectors))
        # Get their corresponding target labels
        self.supportVectorTargets = y[lagrangeIndexes]
        print('And their corresponding labels {}\n'.format(self.supportVectorTargets))

        
        # Calculate intercept with first support vector
        self.intercept = self.supportVectorTargets[0]
        for i in range(len(self.lagrMultipliers)):
            supportVectorsThroughKernel = self.kernel(self.supportVectors[i], self.supportVectors[0])
            self.intercept -= self.lagrMultipliers[i] * self.supportVectorTargets[i] * supportVectorsThroughKernel

    def predict(self, X):
        yPred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagrMultipliers)):
                supportVecKernelVal = self.kernel(self.supportVectors[i], sample)
                prediction += self.lagrMultipliers[i] * self.supportVectorTargets[i] * supportVecKernelVal
            prediction += self.intercept
            yPred.append(np.sign(prediction))
            
        return np.array(yPred)
    

    
    
    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plotIn2D(self, X, y=None, labels = []):
        X_transformed = PCA().transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
        plt.legend(handles = graphLegend)
        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plotIn3D(self, X, y=None, labels = []):
        X_transformed = PCA().transform(X, n_components=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
        plt.legend(handles = graphLegend)

        # Do I add the 
        
        plt.show()
     
        
'''
For easy testing
'''

def main():
    
    # Testing functionality with the iris data for now.
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    # Make the targets in range -1 and 1
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    svm = SupportVectorMachine()
    svm.fit(X_train, y_train)
    yPred = svm.predict(X_test)
    
    print ("Accuracy:", accuracy_score(y_test, yPred))
    
    svm.plotIn2D(X_test, yPred, ['','',''])
#    svm.plotIn2D(X_test, yPred, features = [0,1,2,3], featureLabels = ['a','b'])


if __name__ == "__main__":
    main()
