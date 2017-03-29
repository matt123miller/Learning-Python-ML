# -*- coding: utf-8 -*-
  
import sys
import os
import math
import random
from sklearn import datasets
import numpy as np
from point import Point

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import normalize
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA


class KMeansClustering():
    def __init__(self, k=2, maxIterations=500):
        self.k = k
        self.maxIterations = maxIterations

    # Initialize the centroids as random samples
    def _initRandomCentroids(self, X):
        nSamples, nFeatures = np.shape(X)
        # Preallocate an array of the X shape
        centroids = np.zeros((self.k, nFeatures))
        # A random item in X will be chosen as a centroid for each K cluster
        for i in range(self.k):
            centroids[i] = X[np.random.choice(range(nSamples))]
        return centroids

    # Return the index of the closest centroid to the sample
    def _closestCentroid(self, sample, centroids):
        # i is the row in X
        closest_i = None
        closestDistance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = Point.distance(sample, centroid)
            if distance < closestDistance:
                closest_i = i
                closestDistance = distance
        return closest_i

    # Assign the samples to the closest centroids to create clusters
    def _createClusters(self, centroids, X):
        nSamples = np.shape(X)[0]
        #Creates a 2d array with 1 row and k columns
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closestCentroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # Calculate new centroids as the means of the samples
    # in each cluster
    def _calculateCentroids(self, clusters, X):
        nFeatures = np.shape(X)[1]
        centroids = np.zeros((self.k, nFeatures))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # Classify samples as the index of their clusters
    def _getClusterLabels(self, clusters, X):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Do K-Means clustering and return cluster indices
    def predict(self, X):
        # Initialize centroids
        centroids = self._initRandomCentroids(X)

        # Iterate until convergence or for max iterations
        for i in range(self.maxIterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self._createClusters(centroids, X)
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculateCentroids(clusters, X)

            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                print('KMeans converged after {} iterations'.format(i))
                break
        print('Cluster shape: {}'.format(np.shape(clusters)))
#        print('Clusters: {}'.format(clusters))
        return self._getClusterLabels(clusters, X)


def main():
    # Load the dataset
#    X, y = datasets.make_blobs()
    
    data = datasets.load_iris()
#    X = normalize(data.data)
    X = data.data
    y = data.target
    
    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X)
    
    # Project the data onto the 2 primary principal components
    pca = PCA()
    pca.plot_in_2d(X, y_pred, 'Predicted clusters')
    pca.plot_in_2d(X, y, 'Defined clusters')


if __name__ == "__main__":
    main()

