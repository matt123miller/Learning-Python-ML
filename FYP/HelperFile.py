# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg 
from point import Point

class Helper():

    @staticmethod
    def splitDataAboveBelowMean(npIn, returnType):
        above = np.array([])
        below = np.array([])
        
        if returnType == 'm':
            mean = np.mean(npIn)
            above = npIn[npIn > mean]
            below = npIn[npIn < mean]
        else: #it's for points
            mean = Point.averagePoints(npIn).sqrMagnitude()
#            for p in npIn:
#                sqrMag = p.sqrMagnitude()
#                if sqrMag > mean:
#                    above = np.append(above, p)
#                else:
#                    below = np.append(below, p)
            
            above = [p for p in npIn if p.sqrMagnitude() > mean]
            below = [p for p in npIn if p.sqrMagnitude() < mean]
        return above, below
   
    
    '''
    Definitely refactor into Participant, 
    In a loop do the first part computing rfrom and rto and pass the testB participant as argument
    It will be in line with other single participant graphs and keep main clean.
    '''
    @staticmethod
    def graphParticipantsAboveBelow(participants, pCount, trial, labels):
        # 0 for trial 1a and 1b, 2 for 2a and 2b, 4 for 3a and 3b
        trials = [0,0,2,4]
        rfrom = pCount * trials[trial]
        rto = rfrom + pCount
        
        for i in range(rfrom, rto):
            testA = participants[i]
            testB = participants[i + pCount] # This will fetch the same participants alternate movement test.
            
            
            aX = [cp.x for cp in testA.normalisedAboveMean]
            aY = [cp.y for cp in testA.normalisedAboveMean]
            bX = [cp.x for cp in testB.normalisedAboveMean]
            bY = [cp.y for cp in testB.normalisedAboveMean]
             
            ''' Labels and things '''
            testAName = testA.name.split()[1]
            testBName = testB.name.split()[1]
            pName = testA.name.split()[2].upper()
            title = 'CoP values during extension for\n{} (green) vs {} (red) for participant {}'.format(testAName, testBName, pName)
           
         
            ''' Graph all the things '''
            plt.scatter(aX, aY, color = 'g')
            plt.scatter(bX, bY, color = 'r')
    #        plt.xlim([-1,3])
    #        plt.ylim([-6,6])
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(title)
            
            plt.show()
        
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
    
    def constructSmallDataBundle(P, key = 'cop'):
                
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        target = 0
        dataRows = []
        targets = []
        
        #if contains a pendulum trial then targets is 0, hinge trials target is 1
        if any(s in P.name.lower() for s in targetPendulumNames):
            target = 1
            
        if key.lower() == 'cop':
                  
            xed = [cp.x for cp in P.extensionDifferences]
            yed = [cp.y for cp in P.extensionDifferences]
            
            dataRows = []
            targets = []
    
            for i in range(len(P.extensionDifferences)):
                item = [ xed[i].item(), yed[i].item() ] 
                dataRows.append(item)
                targets.append(target)
                

    #    elif key.lower() == 'data6':
    #       Not supporting this yet
        
        return {'data':np.array(dataRows).astype(float), 
                'target':np.array(targets).astype(int), 
                'data_feature_names':np.array(['xExtDiff', 'yExtDiff']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
        
            

    @staticmethod
    def constructDataBundle(P, key = 'cop'):
                
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        target = 0
        dataRows = []
        targets = []
        
        #if contains a pendulum trial then targets is 0, hinge trials target is 1
        if any(s in P.name.lower() for s in targetPendulumNames):
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
        
    