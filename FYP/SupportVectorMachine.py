# -*- coding: utf-8 -*-
''' Included libraries'''
import math
import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxopt
from sklearn import datasets # Used as a test in main below
#from sklearn import 

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
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        
        
        print('My SVM reporting for duty!')
        pass
    
    
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
        
        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])
        
        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')
        
        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
        
        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])
        
        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]
        
        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
    
    def transform(self, X, n_components):
        covariance = calculate_covariance_matrix(X)

        # Get the eigenvalues and eigenvectors.
        # (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed
     # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y = None, features = [], featureLabels = []):
        X_transformed = self.transform(X, n_components=4)
        f1, f2 = features[0], features[1]

        x1 = X_transformed[:, f1]
        x2 = X_transformed[:, f2]
        plt.scatter(x1, x2, c=y)
        plt.xlabel(featureLabels[f1])
        plt.ylabel(featureLabels[f2])
        
        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y = None, features = [], featureLabels = []):
        X_transformed = self.transform(X, n_components=4)
        f1, f2, f3 = features[0], features[1], features[2]
        
        x1 = X_transformed[:, f1]
        x2 = X_transformed[:, f2]
        x3 = X_transformed[:, f3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        ax.set_xlabel(featureLabels[f1])
        ax.set_ylabel(featureLabels[f2])
        ax.set_zlabel(featureLabels[f3])
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
    pca.plot_in_2d(X_test, y_pred, features = [0,1])


if __name__ == "__main__":
    main()
