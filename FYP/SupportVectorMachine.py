# -*- coding: utf-8 -*-
''' Included libraries'''
import math
import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxopt
from sklearn import datasets # Probably not used

''' My code '''
from participant import Participant

''' OS stuff '''
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, normalize
from data_operation import accuracy_score
from kernels import *
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class SupportVectorMachine(object):
    def __init__(self, kernel, C = 0, power = 4, gamma = None, coef = 4):
        '''
        Kernel is the 
        C is the hyperparameter that controls the effect of the soft margin. I assume for now that 0 means no soft margin.
        '''
        self.C = C 
        
        
        print('My SVM reporting for duty!')
        pass
    
    
    
    
    
     # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y = None, features = []):
        X_transformed = PCA.transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y = None, features = [], featureLabels = []):
        X_transformed = self.transform(X, n_components=3)
        f1, f2, f3 = features[0], features[1], features[2]
        
        x1 = X_transformed[:, f1]
        x2 = X_transformed[:, f2]
        x3 = X_transformed[:, f3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.xlabel(featureLabels[f1])
        plt.ylabel(featureLabels[f2])
        plt.zlabel(featureLabels[f3])
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
    
    clf = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print ("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)


if __name__ == "__main__":
    main()
