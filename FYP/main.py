# -*- coding: utf-8 -*-

# Libraries
import sys
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from enum import Enum

# My code
from point import Point
from HelperFile import Helper
from SupportVectorMachine import SupportVectorMachine as MySVM
from KMeansClustering import KMeansClustering as KMeans
from participant import Participant
from kernels import *


# Some helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../utils")
from data_manipulation import train_test_split, shuffle_data, normalize
from data_operation import accuracy_score
#from kernels import *

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
pCount = len(participantNames)
tCount = len(trialNames)
byValue = 15
threshold = 0.5
returnType = 'p'


class MLType(Enum):
    SVM = 0
    KMEANS = 1
    # Add more types as necessary



''' Choose what algorithm you wanna do '''
chosenAlgorithm = MLType.SVM
    

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
          'data_feature_names':np.array([ 'Depends on what's stored in the data array'])}
'''

def loadParticipants(trials, names):
    outParticipants = []

    for trial in trials:
        for name in names:
            p = Participant(name = date + ' ' + trial + ' ' + name, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants


#def createParticipantFeatures(participants):

    '''
    This loop is what creates all the data required for later steps
    '''
#    for p in participants[:]:
        
    

    
def main():
    
    
    if chosenAlgorithm == MLType.KMEANS:
        print("We're doing K-Means Clustering!")
    elif chosenAlgorithm == MLType.SVM:
        print("We're doing SVM")
    
    # pass this to all methods
    # use a slice on trialnames or participant names!
    participants = loadParticipants(trials = trialNames[0:tCount], names = participantNames[0:pCount])
    
  
    print('The plateaus were computed by looking {0} values ahead and saving values below {1}'.format(byValue, threshold))

    '''
    Create my features
    '''
    for p in participants:
        p.generateFeatures(byValue, threshold)


#    Maybe we should just deal with there being some participants having less than 10 above and belows? Will it matter?
#    participants = [p for p in participants if len(p.aboveMean) + len(p.belowMean) >= 20]
    
    # Validate that the above and below mean data is the same length - Ideally 10 values.
    p = participants[-8]
    print('above mean is {} long: {}'.format(len(p.aboveMean),p.aboveMean))
    print('below mean is {} long: {}'.format(len(p.belowMean),p.belowMean))
    print(p.meanPoint)
    return


    '''
    MAYBE
    Introduce a loop here that will create graphs and whatnot out of each individual Participant
    Much cleaner
      
        for i, obj in enumerate(whatever):
            do things
            
    When I refactor graphParticipantsAboveBelow into the Participant object make the loop I mentioned in Helper here.
    '''     
    

    '''
    Shows each participants difference between extension values in a and b tests
    '''
#    Helper.graphParticipantsAboveBelow(participants, pCount, trial = 2, labels = [xCopLabel, yCopLabel])
#    Helper.saveFigures(participants, 'x & y over time', xCopLabel, yCopLabel)
#    return


    '''
    Try to bruteforce feature selection here?
    sklearn.feature_selection 
    chi squared
    make many bundles using a 2 loops to combine every combo of features
    '''

    # create an array out of each participants features and loop that that to create bundles
    



    
    print('All data manipulation is hopefully done now \nNow to make graphs and things out of each participant')



    
    ''' different slices '''
    beginOne, endOne = 0, pCount * 2
    beginTwo, endTwo = pCount * 2, pCount * 4
    beginThree, endThree = pCount * 4, pCount * 6

    chosenSlice = participants[beginOne:endOne]

    
    
    '''
    RETURNING HERE WHILE I GRAPH RANDOM THINGS TO TRY AND FIND THE RIGHT DATA FEATURES
    '''
    return
    

    '''
    This section MUST BE CHANGED to work with a chosen subset of features.
    
    '''


    
    ''' Create each bundle using the chosenSlice '''
    bundles = []
    
    if chosenAlgorithm == MLType.KMEANS:
        
        for participant in chosenSlice[:pCount]: #First half
            
            chosenData = participant.extensionLengths
                
            bundle = Helper.constructSmallDataBundle(participant.name, chosenData, target = 0, key = 'cop')
            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)
        
        ''' 
        Is it a good idea to make more bundles with different chosenData before stacking them? 
        Lets try
        '''
        
        for participant in chosenSlice[pCount:]: #Second half
        
            chosenData = participant.extensionLengths
            
            bundle = Helper.constructSmallDataBundle(participant.name, chosenData, target = 1, key = 'cop')
            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)   
    
    
    elif chosenAlgorithm == MLType.SVM:
    
        for participant in chosenSlice[:]: # Whole slice
            
            chosenData = participant.extensionLengths
                
            bundle = Helper.constructDataBundle(participant, key = 'cop')
            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)
    
    ''' 
    Create a big bundle combining all the individual bundles
    '''
    
    bigBundle = Helper.appendDataBundles(bundles[:])

     # Load the dataset
    
#    X = normalize( bigBundle['data'], 0 ) # Normalising over the 0th axis seems good. The 1st axis is awful though

    X = bigBundle['data']
    y = bigBundle['target']
    
    
    #Quick graph test of data
#    plt.scatter(X[:,0], X[:,1], c='g') 
##    plt.scatter(X[:,2], X[:,3], c='r') 
##
#    plt.show()
#    return

    '''
    The targets define whether an element belongs to a class (1) or not (-1)
    Important for SVM and should be False
    KMeans doens't matter, I've set to True but I don't think it matters
    I've actually moved the code for this into the if statements as each algorithm requires it's own thing.
    '''
#    flipTargets = True
#    
#    if flipTargets:
#    else:
        
    
    
    
    
    
    
    if chosenAlgorithm == MLType.KMEANS:
    
        ''' 
        Trying some KMeans because I can see clusters myself and I'm desperate for something
        '''
        print('The target values are flipped to be 0 for hinge movement or 1 for pendulum movement')

        # Cluster the data using K-Means
        kmeans = KMeans(k=2)
        yPred = kmeans.predict(X)
        
        '''
        Green points will hopefully show pendulum movement, red for hinge
        '''
        colours = ['green' if l == 1 else 'red' for l in y]
        predColours = ['green' if l == 1 else 'red' for l in yPred]
    
        # Project the data onto the 2 primary principal components
        kmeans.plot_in_2d(X, predColours, ['Predicted clusters for leaning right extension lengths', xCopLabel, yCopLabel])
        kmeans.plot_in_2d(X, colours, ['Defined clusters for leaning right extension lengths', xCopLabel, yCopLabel])
     
        
    elif chosenAlgorithm == MLType.SVM:
        
        '''
        Make an SVM out of ALL with bigbundle or a SUBSET OF participants data using bundles?
        Maybe pass a slice of bundles to appendDataBundles
        '''
        
        y[y == 0] = -1
        print('Target values are the default -1 for hinge movement or 1 for pendulum movement')
     
        ''' 
        Data is all read and processed sequentially by file up until this point
        The train_test_split method shuffles the data before splitting it
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #
        print('xtrain length {} \nytrain length {} \nxtest length {}\nytest length {}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))
        
        # What kernel would you like to use? Make sure I'm importing the right kernel file.
        chosenKernel = linear_kernel
        svm = MySVM(kernel=chosenKernel, C = 1, power=4, coef=2)
        
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
    
        print ("Kernel: {}, Accuracy: {}".format(chosenKernel, accuracy_score(y_test, y_pred)))
    
        colours = ['green' if l == 1 else 'red' for l in y_test]

        # Reduce dimensions and plot the results
        svm.plot_in_2d(X_test, colours, ['',xCopLabel,yCopLabel])
    #    

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
        
        
#        svm = MySVM(kernel=polynomial_kernel, power=4, coef=1)
#        svm.fit(X_train, y_train)
#        y_pred = svm.predict(X_test)
#    
#        print ("Accuracy:", accuracy_score(y_test, y_pred))
#    
#        # Reduce dimension to two using PCA and plot the results
#        svm.plot_in_2d(X_test, y_pred)
    


'''
Compound scatter and line graphs for rest and extension -
A bit old and can probably be made better using the updates to Participant
'''
        
#        if returnType == 'p':
#            p.plotCopHighLows()
#        
#            highMeans = np.append(highMeans, Point.averagePoints(p.aboveMean))
#            lowMeans = np.append(lowMeans, Point.averagePoints(p.belowMean))
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
#        p.plotAvgHighLows(avgPlateaus, show = True)
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