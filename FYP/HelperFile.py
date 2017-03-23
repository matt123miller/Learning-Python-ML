# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg 
from point import Point

class Helper():

    @staticmethod
    def averagePoints(points):
        x, y = [], []
        for p in points:
            x.append(p.x)
            y.append(p.y)
        return Point(sum(x)/len(x),sum(y)/len(y))
    
    @staticmethod
    def splitDataAboveBelowMean(npIn, returnType):
        above = np.array([])
        below = np.array([])
        
        if returnType == 'm':
            mean = np.mean(npIn)
            above = npIn[npIn > mean]
            below = npIn[npIn < mean]
        else: #it's for points
            mean = Helper.averagePoints(npIn).sqrMagnitude()
#            for p in npIn:
#                sqrMag = p.sqrMagnitude()
#                if sqrMag > mean:
#                    above = np.append(above, p)
#                else:
#                    below = np.append(below, p)
            
            above = [p for p in npIn if p.sqrMagnitude() > mean]
            below = [p for p in npIn if p.sqrMagnitude() < mean]
        return above, below
    
    @staticmethod
    def pointListMinusPoint(points, point):
        rlist = []
        for p in points:
             rlist.append(p - point)
        return np.array(rlist)
    
    @staticmethod
    def normaliseOverHighestValue(values):
        outValues = []
        highest = np.max(values)
#        for v in values:
#            outValues.append(v / highest)
        return np.array([v / highest for v in values]) 
    
    # Fit the dataset to the number of principal components
    # specified in the constructor and return the transform dataset
    @staticmethod
    def transform(self, X, n_components):
        covariance = calculate_covariance_matrix(X)

        # Get the eigenvalues and eigenvectors.
        # (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed
     
    @staticmethod
    def constructDataBundle(P, key = 'cop'):
                
        targetHingeNames = ('trial1b', 'trial2b', 'trial3b')
        target = 0
        dataRows = []
        targets = []
        
        #if contains a pendulum trial then targets is 0, hinge trials target is 1
        if any(s in P.name.lower() for s in targetHingeNames):
            target = 1
        
        if key.lower() == 'cop':
              
            length = min(len(P.aboveMean), len(P.belowMean))
                
            xBelow = [cp.x for cp in P.belowMean]
            yBelow = [cp.y for cp in P.belowMean]
            xAbove = [cp.x for cp in P.aboveMean]
            yAbove = [cp.y for cp in P.aboveMean]
            
            dataRows = []
            targets = []
            
            for i in range(length):
                item = [ xBelow[i].item(), yBelow[i].item(), xAbove[i].item(), yAbove[i].item() ]
                dataRows.append(item)
                targets.append(target)
            
    #    elif key.lower() == 'data6':
    #       Not supporting this yet
        
        return {'data':np.array(dataRows).astype(float), 
                'target':np.array(targets).astype(int), 
                'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
            
        
    @staticmethod
    def appendDataBundles(bundles):
        
        outData = np.concatenate([b['data'] for b in bundles]).astype(float)
        outTargets = np.concatenate([b['target'] for b in bundles]).astype(int)
    
        return {'data':outData, 
                'target':outTargets, 
                'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
        
    