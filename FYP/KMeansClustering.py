# -*- coding: utf-8 -*-
  
import sys
import os
import math
import random
import numpy as np
from point import Point

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # This will make the legends of graphs.
import matplotlib.cm as cm # Colormaps are cool

# Import some helper functions
pathOfCurrentDirectory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, pathOfCurrentDirectory + "/../utils")
from principal_component_analysis import PCA

redPatch = mpatches.Patch(color='red', label='Hinge movement')
greenPatch = mpatches.Patch(color='green', label='Pendulum movement')
graphLegend = [greenPatch, redPatch]
chosenCmap = 'magma'

class KMeansClustering():
    def __init__(self, k=2, maxIterations=500):
        self.k = k
        self.maxIterations = maxIterations

    # Initialize the centroids as random samples from all the points
    def createRandomClusters(self, X):
        nSamples, nFeatures = np.shape(X)
        # Initialise an array of equal shape to {K,X}
        centroids = np.zeros((self.k, nFeatures))
        # A random index in X will be chosen to become the first centroid for each cluster K
        for i in range(self.k):
            centroids[i] = X[np.random.choice(range(nSamples))]
        return centroids

    # Find index of the closest centroid to the provided sample
    def closestCentroidToExample(self, sample, centroids):
        # i is the row in X
        closestIndex = None
        closestDistance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = Point.distance(sample, centroid)
            if distance < closestDistance:
                closestIndex = i
                closestDistance = distance
        return closestIndex

    # Assign the samples to the closest centroids, this becomes the clusters
    def createClustersFromCentroids(self, centroids, X):
        nSamples = np.shape(X)[0]
        #Creates a 2d array with 1 row and k columns
        clusters = [[] for _ in range(self.k)]
        for i, sample in enumerate(X):
            centroidForIndex = self.closestCentroidToExample(sample, centroids)
            clusters[centroidForIndex].append(i)
        return clusters

    # Calculate new centroids as the mean of all samples found in each cluster
    def calculateCentroidsFromClusters(self, clusters, X):
        nFeatures = np.shape(X)[1]
        centroids = np.zeros((self.k, nFeatures))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # Classify the provdied samples as the index of their clusters
    def getLabelsOfClusters(self, clusters, X):
        # One prediction for each sample
        yPred = np.zeros(np.shape(X)[0])
        for i, cluster in enumerate(clusters):
            for j in cluster:
                yPred[j] = i
        return yPred

    # Do K-Means clustering and return cluster indices
    def predict(self, X):
        print(np.shape(X))
        # Initialize centroids
        centroids = self.createRandomClusters(X)

        # Iterate until convergence or for max iterations
        for i in range(self.maxIterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self.createClustersFromCentroids(centroids, X)
            previousCentroids = centroids
            # Calculate new centroids from the clusters
            centroids = self.calculateCentroidsFromClusters(clusters, X)

            # If no centroids have changed then it's reached convergence
            diff = centroids - previousCentroids
            if not diff.any():
                print('KMeans converged after {} iterations'.format(i))
                break
        print('Cluster shape: {}'.format(np.shape(clusters)))
#        print('Clusters: {}'.format(clusters))
        return self.getLabelsOfClusters(clusters, X)


    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plotIn2D(self, X, y=None, k = 1, labels = []):
        X_transformed = PCA().transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y, cmap = chosenCmap)
        
        # Plot a star for each centroid
        for i in range(k):
            plt.scatter(np.mean(X_transformed[y == i, 0]), np.mean(X_transformed[y == i, 1]), s = 400, marker = '*', c='w', cmap = chosenCmap)
                

        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
#        plt.legend(handles = graphLegend)
        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plotIn3D(self, X, y=None, k=1, labels = []):
        X_transformed = PCA().transform(X, n_components=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        
        # Plot a star for each centroid
        for i in range(k):
            ax.scatter(np.mean(X_transformed[y == i, 0]), np.mean(X_transformed[y == i, 1]), np.mean(X_transformed[y == i, 2]), s = 400, marker = '*', c='w', cmap = chosenCmap)
           
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
#        plt.legend(handles = graphLegend)

        # Do I add the 
        
        plt.show()
        
def main():
    # Load the dataset
    data = datasets.load_iris()
    
    # Is it worth normalising the data? Probably...
#    X = normalize(data.data)
    X = data.data
    print(np.shape(X))
    y = data.target
    k = 3
    
    # Cluster the data using K-Means
    kmeans = KMeansClustering(k=k)
    yPred = kmeans.predict(X)
    # Plot that shit
    kmeans.plotIn2D(X, yPred, k, ['Predicted clusters','',''])
    kmeans.plotIn2D(X, y, k, ['Defined clusters','',''])
    
    # This is the sklearn stuff.
#    kmeans = KMeans(k)
#    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
#    print(np.shape(X_train))
#    predictedClusters = kmeans.predict(X_test)
#    print(kmeans.score(X_test, yPred))
    

if __name__ == "__main__":
    main()

