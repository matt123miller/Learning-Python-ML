# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy import linalg 
from point import Point

pCount = 12
beginOne, endOne = 0, pCount * 2
beginTwo, endTwo = pCount * 2, pCount * 4
beginThree, endThree = pCount * 4, pCount * 6
    
class Helper():
    
    
    def constructSmallDataBundle(pname, desiredData = [], target = None, key = 'cop'):
                
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        dataRows = []
        targets = []
        
        if target == None:
            #if contains a pendulum trial then targets is 0, hinge trials target is 1
            if any(s in pname.lower() for s in targetPendulumNames):
                target = 1
            
        if key.lower() == 'cop':
                  
#            x = [cp.x for cp in desiredData]
#            y = [cp.y for cp in desiredData]
#            
            dataRows = []
            targets = []
    
            for i, xy in enumerate(desiredData):
                item = [ xy.x.item(), xy.y.item() ] 
                dataRows.append(item)
                targets.append(target)
                

    #    elif key.lower() == 'data6':
    #       Not supporting this yet
        
        return {'data':np.array(dataRows).astype(float), 
                'target':np.array(targets).astype(int), 
                'data_feature_names':np.array(['xExtDiff', 'yExtDiff']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
        
            
        
    @staticmethod
    def constructVariedDataBundle(pname, data = [], targets = 0, key = 'cop'):
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        target = 0
        dataRows = []
        target = []
        
        shape = np.shape(data)
        print(shape)
        rows = shape[0]
        # unknown amount of columns as they are of various lengths
        items = []
        for i, rowContents in enumerate(data):
            x = [cp.x for cp in rowContents]
            y = [cp.y for cp in rowContents]
#            items
#            row = []
#            for point in rowContents:
##                row.append(point.x.item(), point.y.item()) 
#                dataRows.append(row)               
#            target.append(targets)
            
        
    
        return {'data':np.array(dataRows).astype(float), 
                'target':np.array(target).astype(int), 
                'data_feature_names':np.array(['xExtDiff', 'yExtDiff']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
                    
    @staticmethod
    def constructRestExtDataBundle(P, key = 'cop'):
                
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        target = 0
        dataRows = []
        targets = []
        
        #if contains a pendulum trial then targets is 0, hinge trials target is 1
        if any(s in P.trialName.lower() for s in targetPendulumNames):
            target = 1
        
        if key.lower() == 'cop':
              
            length = min(len(P.extensionPoints), len(P.restPoints))
                
            xBelow = [cp.x for cp in P.restPoints]
            yBelow = [cp.y for cp in P.restPoints]
            xAbove = [cp.x for cp in P.extensionPoints]
            yAbove = [cp.y for cp in P.extensionPoints]
            
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
    def constructBigDataBundle(P, key = 'cop'):
                
        targetPendulumNames = ('trial1a', 'trial2a', 'trial3a')
        target = 0
        dataRows = []
        targets = []
        
        #if contains a pendulum trial then targets is 0, hinge trials target is 1
        if any(s in P.trialName.lower() for s in targetPendulumNames):
            target = 1
        
        if key.lower() == 'cop':
              
            length = min(len(P.extensionPoints), len(P.restPoints))
                
            xBelow = [cp.x for cp in P.restPoints]
            yBelow = [cp.y for cp in P.restPoints]
            xAbove = [cp.x for cp in P.extensionPoints]
            yAbove = [cp.y for cp in P.extensionPoints]
            betweenX = [cp.x for cp in P.vectorsBetween]
            betweenY = [cp.y for cp in P.vectorsBetween]
            anglesBetween = [a for a in P.anglesBetween]

            
            for i in range(length):
                item = [ xBelow[i].item(), yBelow[i].item(), xAbove[i].item(), yAbove[i].item(), 
                        betweenX[i].item(), betweenY[i].item(), anglesBetween[i].item() ]
                dataRows.append(item)
                targets.append(target)
            
        
        return {'data':np.array(dataRows).astype(float), 
                'target':np.array(targets).astype(int), 
                'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove', 'xBetween', 'yBetween', 'anglesBetween']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
        
    @staticmethod
    def appendDataBundles(bundles):
        
        outData = np.concatenate([b['data'] for b in bundles]).astype(float)
        outTargets = np.concatenate([b['target'] for b in bundles]).astype(int)
    
        return {'data':outData, 
                'target':outTargets, 
                'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove']).astype(str),
                'target_names':np.array(['pendulum', 'hinge']).astype(str)}
        
    
    
    
    @staticmethod
    def saveFigures(participants, testLabel, xCopLabel, yCopLabel):
    
        
        with PdfPages('Test 1a {}.pdf'.format(testLabel)) as pdf:
        
            for p in participants[beginOne:pCount]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')

#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                plt.show()
                pdf.savefig(fig, bbox_inches = 'tight')

        with PdfPages('Test 1b {}.pdf'.format(testLabel)) as pdf:
            
            for p in participants[pCount:endOne]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')
#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                pdf.savefig(fig, bbox_inches = 'tight')
    
        with PdfPages('Test 2a {}.pdf'.format(testLabel)) as pdf:
            
            for p in participants[beginTwo:beginTwo + pCount]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')
#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                pdf.savefig(fig, bbox_inches = 'tight')
                
        with PdfPages('Test 2b {}.pdf'.format(testLabel)) as pdf:
            
            for p in participants[beginTwo + pCount:endTwo]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')
#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                pdf.savefig(fig, bbox_inches = 'tight')
        
        with PdfPages('Test 3a {}.pdf'.format(testLabel)) as pdf:
            
            for p in participants[beginThree:beginThree + pCount]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')
#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                pdf.savefig(fig, bbox_inches = 'tight')
                
        with PdfPages('Test 3b {}.pdf'.format(testLabel)) as pdf:
            
            for p in participants[beginThree + pCount:endThree]:
                fig = plt.figure()
                plt.plot(np.arange(len(p.copX)), [x for x in p.copX])
                plt.plot(np.arange(len(p.copX)), [y for y in p.copY])
                plt.scatter(p.copX[0], p.copY[0], color = 'g')
                plt.scatter(p.copX[-1], p.copY[-1], color = 'r')
#                plt.xlim([0,10])
#                plt.ylim([0,10])
                plt.xlabel(xCopLabel)
                plt.ylabel(yCopLabel)
                plt.title('{} {}'.format(p.name, testLabel))
                pdf.savefig(fig, bbox_inches = 'tight')

    
#    @staticmethod
#    def splitDataAboveBelowMean(npIn, returnType):
#        above = np.array([])
#        below = np.array([])
#        mean = 0
#        
#        if returnType == 'm':
#            mean = np.mean(npIn)
#            above = npIn[npIn > mean]
#            below = npIn[npIn < mean]
#        else: #it's for points
#            mean = Point.averagePoints(npIn).sqrMagnitude()
##            for p in npIn:
##                sqrMag = p.sqrMagnitude()
##                if sqrMag > mean:
##                    above = np.append(above, p)
##                else:
##                    below = np.append(below, p)
#                
#            above = [p for p in npIn if p.sqrMagnitude() > mean]
#            below = [p for p in npIn if p.sqrMagnitude() < mean]
#        return above, below, mean
   
    
    '''
    Definitely refactor into Participant, 
    In a loop do the first part computing rfrom and rto and pass the testB participant as argument
    It will be in line with other single participant graphs.
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
            
            
            aX = [cp.x for cp in testA.extensionLengths]
            aY = [cp.y for cp in testA.extensionLengths]
            bX = [cp.x for cp in testB.extensionLengths]
            bY = [cp.y for cp in testB.extensionLengths]
             
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