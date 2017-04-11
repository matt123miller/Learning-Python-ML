# -*- coding: utf-8 -*-
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

''' My code '''
from participant import Participant

''' OS stuff '''
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, normalize
from data_operation import accuracy_score, calculate_covariance_matrix

from kernels import *
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

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
        self.supportVectorLabels = None
        self.intercept = None
        
        
        print('My SVM reporting for duty!')
        print('')
        pass
    
    
#    def computeLagrangeMultipliers(self, X, y, kernelMatrix, n_samples):
#        # Define the quadratic optimization problem
#        # This shit is gibberish
#        P = cvxopt.matrix(np.outer(y, y) * kernelMatrix, tc='d')
#        q = cvxopt.matrix(np.ones(n_samples) * -1)
#        A = cvxopt.matrix(y, (1, n_samples), tc='d')
#        b = cvxopt.matrix(0, tc='d')
#        
#        if not self.C:
#            G = cvxopt.matrix(np.identity(n_samples) * -1)
#            h = cvxopt.matrix(np.zeros(n_samples))
#        else:
#            G_max = np.identity(n_samples) * -1
#            G_min = np.identity(n_samples)
#            G = cvxopt.matrix(np.vstack((G_max, G_min)))
#            h_max = cvxopt.matrix(np.zeros(n_samples))
#            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
#            h = cvxopt.matrix(np.vstack((h_max, h_min)))
#        
#        # Solve the quadratic optimization problem using cvxopt
#        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
#        
#        # Lagrange multipliers
#        return np.ravel(minimization['x'])
        
    def fit(self, X, y):

        n_samples, n_features = np.shape(X)
        
        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features
        
        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)
        
        # Calculate a matrix of values that came through the kernel
        # Fill it with zeroes first
        kernel_matrix = np.zeros((n_samples, n_samples))
        # Replace each cell with it's values passed through the kernel
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])
        
        lagrMult = self.computeLagrangeMultipliers(X, y, kernel_matrix, n_samples)
        
        print('Lagrange Multipliers {}\n'.format(lagrMult))
        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagrMult > 1e-7 # this means 0.00000001 
        # Get the corresponding lagr. multipliers
        self.lagrMultipliers = lagrMult[idx]
        print('Multipliers > 0 {}\n'.format(self.lagrMultipliers))
        # Get the samples that will act as support vectors
        self.supportVectors = X[idx]
        print('I have {} support vectors {}\n'.format(len(self.supportVectors),self.supportVectors))
        # Get the corresponding labels
        self.supportVectorLabels = y[idx]
        print('And their corresponding labels {}\n'.format(self.supportVectorLabels))

        
        # Calculate intercept with first support vector
        self.intercept = self.supportVectorLabels[0]
        for i in range(len(self.lagrMultipliers)):
            self.intercept -= self.lagrMultipliers[i] * self.supportVectorLabels[
                i] * self.kernel(self.supportVectors[i], self.supportVectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagrMultipliers)):
                prediction += self.lagrMultipliers[i] * self.supportVectorLabels[
                    i] * self.kernel(self.supportVectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    

    
    
    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, labels = []):
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
    def plot_in_3d(self, X, y=None, labels = []):
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
    
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    svm = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    print ("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Reduce dimension to two using PCA.transform and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred, ['','',''])
#    svm.plot_in_2d(X_test, y_pred, features = [0,1,2,3], featureLabels = ['a','b'])


if __name__ == "__main__":
    main()
