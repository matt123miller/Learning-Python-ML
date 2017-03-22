# -*- coding: utf-8 -*-


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from point import Point
from HelperFile import Helper
from SupportVectorMachine import SVMTrainer
from participant import Participant
#from kernels import *


# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, shuffle_data, normalize
from data_operation import accuracy_score
from kernels import *
sys.path.insert(0, dir_path + "/../supervised_learning")
from support_vector_machine import SupportVectorMachine # as SVM
from kernels import *
sys.path.insert(0, dir_path + "/../unsupervised_learning/")
from principal_component_analysis import PCA

"""
variables
"""
plateWidth = 100
plateLength = 100
participantNames = ['ac', 'an', 'gd', 'gp', 'hm', 'lh', 'mb', 'mm', 'te', 'wy', 'xd', 'yz'] # Not case sensitive
trialNames = ['Trial1a','Trial1b','Trial2a','Trial2b','Trial3a','Trial3b'] # not case sensitive
trialDescriptions = ['Leaning right in a pendulum movement', 'Leaning right in a hinge movement','Leaning forward in a pendulum movement', 'Leaning forward in a hinge movement','Leaning backward in a pendulum movement', 'Leaning backward in a hinge movement']
date = "170306"
dataKey = 'data6'
xCopLabel = 'Anteroposterior weight distribution'
yCopLabel = 'Mediolateral weight distribution'

"""
TL = front left = data6[[:,0]] = b
TR = front right = data6[[:,1]] = c
BL = back left = data6[[:,2]] = r
BR = back right = data6[[:,3]] = y
"""

'''
Bundle format

bundle = {'data':np.array([[ 4 float values ]]), 
             'targets':np.array([ 0 or 1 ]), 
             'target_names':np.array(['pendulum', 'hinge']),
             'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove'])}
'''

'''
How can I use a nice CMAP for colours? I saw autumn looks nice
'''

def loadParticipants(trials, names):
    outParticipants = []

    for trial in trials:
        for name in names:
            p = Participant(name = date + ' ' + trial + ' ' + name, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants



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
        
        

def appendDataBundles(bundles):
    
    outData = np.concatenate([b['data'] for b in bundles]).astype(float)
    outTargets = np.concatenate([b['target'] for b in bundles]).astype(int)

    return {'data':outData, 
            'target':outTargets, 
            'data_feature_names':np.array(['xBelow', 'yBelow', 'xAbove', 'yAbove']).astype(str),
            'target_names':np.array(['pendulum', 'hinge']).astype(str)}
    
    
def graphParticipantsAboveBelow(participants, pCount, trial):
    # 0 for trial 1a and 1b, 2 for 2a and 2b, 4 for 3a and 3b
    trials = [0,0,2,4]
    rfrom = pCount * trials[trial]
    rto = rfrom + pCount
    
    for i in range(rfrom, rto):
        testA = participants[i]
        testB = participants[i + pCount] # This will fetch the same participants alternate movement test.
        
        ''' 
        Data 
        Maybe I should normalise all values here (ideally between -1 and 1)
        '''
#        aX = Helper.normaliseOverHighestValue([cp.x for cp in testA.normalisedAboveMean])
#        aY = Helper.normaliseOverHighestValue([cp.y for cp in testA.normalisedAboveMean])
#        bX = Helper.normaliseOverHighestValue([cp.x for cp in testB.normalisedAboveMean])
#        bY = Helper.normaliseOverHighestValue([cp.y for cp in testB.normalisedAboveMean])
        
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
        plt.xlabel(xCopLabel)
        plt.ylabel(yCopLabel)
        plt.title(title)
        
        plt.show()
        
def main():
        
    pCount = len(participantNames)
    tCount = len(trialNames)
    byValue = 15
    threshold = 0.5
    returnType = 'p'

    # Do I want a specific subset of files for some reason?

    # pass this to all methods
    # use a slice on trialnames or participant names!
    participants = loadParticipants(trials = trialNames[0:tCount], names = participantNames[0:pCount])
    
    
    
    highMeans = np.array([])
    lowMeans = np.array([])
    diffMeans = np.array([])
    normalisedAboveMean = np.array([])
    
    print('The plateaus were computed by looking {0} values ahead and saving values below {1}'.format(byValue, threshold))

    '''
    This loop is what creates all the data required for later steps
    '''
    

    for p in participants[:]:
        
        plateaus = p.lookAheadForPlateau(by = byValue, varianceThreshold = threshold)
        
#        p.normaliseData() #As in the vector normalise
        
        avgPlateaus = p.averagePlateauSections(plateaus, returnType)
        
        #returns numpy arrays
        p.aboveMean, p.belowMean = Helper.splitDataAboveBelowMean(avgPlateaus, returnType) 
        p.meanRestPoint = Helper.averagePoints(p.belowMean)
                
        '''
        Now that I've got a somewhat normalised value for each plateau above the 
        mean rest point I can graph each participant for their differences between 
        tests a and b for each direction. Then SVM that to get an actual project?
        '''
        aboveMean = Helper.pointListMinusPoint(p.aboveMean, p.meanRestPoint)
        p.normalisedAboveMean = np.append(normalisedAboveMean, aboveMean)

  
    '''
    Introduce a loop here that will create graphs and whatnot out of each individual Participant
    Much cleaner
      
        for i in whatever:
            do things
    '''     
    
    print('All data manipulation is hopefully done now \nNow to make graphs and things out of each participant')




    '''
    Shows each participants extension values
    '''
#    graphParticipantsAboveBelow(participants, pCount, trial = 2)


    
    bundles = []
    
    for p in participants[:]:
        bundle = constructDataBundle(p, 'cop')
        bundles.append(bundle)
      
        
    '''
    Make an SVM out of ALL participants data.
    '''
    
    bigBundle = appendDataBundles(bundles)
    
    
    X = normalize(bigBundle['data'])
    y = bigBundle['target']
    
    '''
    Will I also have to do this? Is this important for the convex optimisation/lagrange multipliers???
    y[y == 1] = -1
    y[y == 2] = 1
    Lets give it a try and see what happens!
    '''
    flipTargets = True
    
    if flipTargets:
        y[y == 0] = -1
        print('The target values are flipped to be -1 for pendulum movement or 1 for hinge movement')
    else:
        print('Target values are 0 for pendulum movement or 1 for hinge movement')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print('xtrain length {} \nytrain length {} \nxtest length {}\nytest length {}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))
    
    chosenKernel = polynomial_kernel
    clf = SupportVectorMachine(kernel=chosenKernel, power=4, coef=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print ("Kernel: {}, Accuracy: {}".format(chosenKernel, accuracy_score(y_test, y_pred)))

    # Reduce dimension to two using PCA and plot the results
    pca = PCA()
    pca.plot_in_2d(X_test, y_pred)
    

'''
Trying to make an SVM from each participant
'''

#    for bundle in bundles[:2]:
#        X = normalize(bundle['data'])
#        y = bundle['target']
#        print('Bundle normalised data is {}'.format(X))
#        print('X shape is {}'.format(np.shape(X)))
#
#        print('Y targets are {}'.format(y))
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
#        print('X_train is {}\nX_test is {}'.format(X_train, X_test))
#        print('')
        
        
#        clf = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
#        clf.fit(X_train, y_train)
#        y_pred = clf.predict(X_test)
#    
#        print ("Accuracy:", accuracy_score(y_test, y_pred))
#    
#        # Reduce dimension to two using PCA and plot the results
#        pca = PCA()
#        pca.plot_in_2d(X_test, y_pred)
    


'''
Compound scatter and line graphs for rest and extension -
A bit old and can probably be made better using the updates to Participant
'''
        
#        if returnType == 'p':
#            p.plotCopHighLows()
#        
#            highMeans = np.append(highMeans, Helper.averagePoints(p.aboveMean))
#            lowMeans = np.append(lowMeans, Helper.averagePoints(p.belowMean))
#            
#        else:
#            highMean = np.mean(p.aboveMean)
#            lowMean = np.mean(p.belowMean)
#            print('The highs are {} with a mean of {}'.format(p.aboveMean, highMean))
#            print('the lows are{} with a mean of {}'.format(p.belowMean, lowMean))
#            print('') #empty row
#            highMeans = np.append(highMeans, highMean)
#            lowMeans = np.append(lowMeans, lowMean)
        
#        p.compoundScatterLine(plateaus)
#        p.showAvgHighLows(avgPlateaus, show = True)
#        plt.scatter(np.arange(len(avgPlateaus)), avgPlateaus)
   


'''
Finally a section for making any graphics out of ALL data
'''
    
##    if returnType == 'p':
##        # make a graph of the points, cartesian style
##
##        xDataLow = [cp.x for cp in lowMeans]
##        xDataHigh = [cp.x for cp in highMeans]
##        yDataLow = [cp.y for cp in lowMeans]
##        yDataHigh = [cp.y for cp in highMeans]
##        plt.scatter(xDataLow, yDataLow, color = 'g')
##        plt.scatter(xDataHigh, yDataHigh, color = 'r')
##        plt.title('Shows the average CoP values when at rest (in green)\nand extension (in red) for all participants during test 1a')
##        plt.xlabel('CoP values on the x axis')
##        plt.ylabel('CoP values on the y axis')
##        plt.show()
##        
##    elif returnType == 'm':
##        
##        print('{}'.format(highMeans))
##        print('{}'.format(lowMeans))
##        diffMeans = highMeans - lowMeans
##        print(diffMeans)
#        
##    axis = np.arange(len(meanDiffs))
##    plt.scatter(axis, highMeans, color= 'r')
##    plt.scatter(highMeans, lowMeans, color = 'g')
##    plt.show()




if __name__ == "__main__":
    main()