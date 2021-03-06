'''
This file originally came from https://github.com/eriklindernoren/ML-From-Scratch/blob/master/unsupervised_learning/principal_component_analysis.py
It has been adapted a lot and most of the original was removed. This file is basically used only for the Transform method.
'''

import sys
import os
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_operation import calculate_covariance_matrix
from data_operation import calculate_correlation_matrix
from data_manipulation import standardize

redPatch = mpatches.Patch(color='red', label='Hinge movement')
greenPatch = mpatches.Patch(color='green', label='Pendulum movement')
        
graphLegend = [greenPatch, redPatch]

class PCA():
    def __init__(self): pass

    

    def plotInNd(self, features, X, y = None):
        n = len(features)
        X_transformed = self.transform(X, n_components = max(features)+1)
        
        # Another option is to loop and plot a single feature at a time and show at the end?
        
        for i in range(n):
            
            pass
            
            
        if n == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x1 = X_transformed[:, features[0]]
            x2 = X_transformed[:, features[1]]
            x3 = X_transformed[:, features[2]]
            ax.scatter(x1, x2, x3, c=y)
        else:
            x1 = X_transformed[:, features[0]]
            x2 = X_transformed[:, features[1]]
            plt.scatter(x1, x2, c=y)
        plt.show()
        
    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, labels = []):
        X_transformed = self.transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
#        plt.title(labels[0])
#        plt.xlabel(labels[1])
#        plt.ylabel(labels[2])
#        plt.legend(handles = graphLegend)
        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y=None, labels = []):
        X_transformed = self.transform(X, n_components=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
#        plt.title(labels[0])
#        plt.xlabel(labels[1])
#        plt.ylabel(labels[2])
        
        # Do I add the 
        
        plt.show()

    # Fit the dataset to the number of principal components
    # specified in the constructor and return the transform dataset
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

def main():
    # Load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components and plot the
    # data
    pca = PCA()
#    pca.plot_in_3d(X, y, ['','',''])
    pca.plot_in_3d( X, y, ['','',''])

if __name__ == "__main__":
    main()
