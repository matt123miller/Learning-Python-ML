# -*- coding: utf-8 -*-

# Libraries
import sys
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from enum import Enum
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif as anova
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, cross_val_predict, ShuffleSplit


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
from decision_tree import ClassificationTree



class MLType(Enum):
    SVM = 0
    KMEANS = 1
    # Add more types as necessary


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
byValue = 25
threshold = 0.45
returnType = 'p'


''' Choose what algorithm you wanna do '''
chosenAlgorithm = MLType.KMEANS
    

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
            p = Participant(date, name, trial, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants

 
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
    for p in participants[:]:
        p.generateFeatures(byValue, threshold)


    '''
    Try to bruteforce feature selection here?
    sklearn.feature_selection 
    chi squared
    make many bundles using a 2 loops to combine every combo of features
    '''
 

    # This is the next step! Prepare to brute force it.    
    # Should this go before or after the following bundle section? Should it replace it? Who knows.

    # Here's the plan I wrote out
    
    # Loop through 3 movement directions
        # Make bundles containing both movement types, given targets appripriate for ML type.
        
        # Somehow loop to make every combination of features, somehow include the feature names of the combo
            # That's probably nested x y style
                # Make KMeans for the combo
                # Make SVM for the combo
                # Measure the accuracy/performance and record that and the name of the combo
                    # This can probably be saved to a dict here, keys are the combo name, values are the performance
                
    # The dict recording performance can later be translated into a csv file for uploading as an appendix to the project.
    
    ''' different index slices '''
    beginOne, endOne = 0, pCount * 2
    beginTwo, endTwo = pCount * 2, pCount * 4
    beginThree, endThree = pCount * 4, pCount * 6
    
    directionSlices = [[beginOne, endOne], [beginTwo, endTwo], [beginThree, endThree]]
    featureLabels = []
    dataInstances = []
    directionBundles = []
    
    for dSlice in directionSlices:
        
        participantbundleforslice = {'data':[],
                                     'target':[],
                                     'featureLabels':[]
                                     }
        
        for p in participants[dSlice[0] : dSlice[1]]:
            listDict = p.listFeaturesDict()
            
            features = listDict['features']
            labels = listDict['featureNames']
            flength = len(features[0])
            
            for i in range(flength):
                dataRow = []
                for value in features:
                    dataRow.append(value[i])
                    
                participantbundleforslice['data'].append(dataRow)
                
                # What target does this row 
                if chosenAlgorithm == MLType.SVM:
                    if p.movement == 'pendulum':
                        participantbundleforslice['target'].append(1)
                    if p.movement == 'hinge':
                        participantbundleforslice['target'].append(-1) 
                                                 
                elif chosenAlgorithm == MLType.KMEANS:
                    if p.movement == 'pendulum':
                        participantbundleforslice['target'].append(1)
                    if p.movement == 'hinge':
                        participantbundleforslice['target'].append(0)  
       

        directionBundles.append(participantbundleforslice)
        # Validate for myself.
        print(np.shape(participantbundleforslice['data'])) 
        print(participantbundleforslice['target'])     
        
        print(participantbundleforslice['data'][100][5]) # You can access the data like this, 'data', row (instance), column (feature) 
    # directionBundles should now contain 3 dictionaries,
    # each one containing all rows of data for a direction and a target value dependent on movement type.
    
    return


#    bundleMatrixOfLists = {}
#    bundleMatrixOfSingleValues = {}
#
#    for p in participants[:]:
#        featuresDict = p.namesAndListFeatures()
#        key = p.participantName + ' ' + p.trialName
#        bundleMatrixOfLists[key] = featuresDict
#        bundleMatrixOfSingleValues[key] = p.namesAndSingleFeatures()
#    
#    print(bundleMatrixOfLists['mm Trial1a']['plateauSensorAverages']) 
        
#    return 
    print('All data manipulation is hopefully done now. \nNow to make graphs and things out of each participant. \n \n ##########')

    return

    
    

    chosenSlice = participants[beginOne:endOne]

    
    ''' Create each bundle using the chosenSlice '''
    bundles = []
    
    if chosenAlgorithm == MLType.KMEANS:
        
        for participant in chosenSlice[:pCount]: #First half
            
            chosenData = participant.vectorsBetween
                
#            bundle = Helper.constructSmallDataBundle(participant.trialName, chosenData, target = 0, key = 'cop')
            bundle = Helper.constructBigDataBundle(participant)
#            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)
        
        ''' 
        Is it a good idea to make more bundles with different chosenData before stacking them? 
        Lets try
        '''
        
        for participant in chosenSlice[pCount:]: #Second half
        
            chosenData = participant.vectorsBetween
            
#            bundle = Helper.constructSmallDataBundle(participant.fileName, chosenData, target = 1, key = 'cop')
            bundle = Helper.constructBigDataBundle(participant)
#            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)   
    
    
    elif chosenAlgorithm == MLType.SVM:
    
        for participant in chosenSlice[:]: # Whole slice
            
            chosenData = participant.vectorsBetween
                
            bundle = Helper.constructBigDataBundle(participant, key = 'cop')
#            print('length {} vs bundle length {}'.format(len(chosenData), np.shape(bundle['data'])))
            bundles.append(bundle)
    
    ''' 
    Create a big bundle combining all the individual bundles
    '''
    
    bigBundle = Helper.appendDataBundles(bundles[:])

     # Load the dataset
    
#    X = normalize( bigBundle['data'], 0 ) # Normalising over the 0th axis seems good. The 1st axis is awful though

    X = bigBundle['data']
    y = bigBundle['target']

    '''
    The targets define whether an element belongs to a class (1) or not (-1)
    Important for SVM and should be False
    KMeans doens't matter, I've set to True but I don't think it matters
    I've actually moved the code for this into the if statements as each algorithm requires it's own thing.
    '''

    
    
    if chosenAlgorithm == MLType.KMEANS:
    
        ''' 
        Trying some KMeans because I can see clusters myself and I'm desperate for something
        '''
        print('The target values are flipped to be 0 for hinge movement or 1 for pendulum movement')

        # Cluster the data using K-Means
        K = 2
        kmeans = KMeans(k = K)
        yPred = kmeans.predict(X)
        
        '''
        Green points will hopefully show pendulum movement, red for hinge
        '''
        colours = ['green' if l == 1 else 'red' for l in y]
        predColours = ['green' if l == 1 else 'red' for l in yPred]
    
        # Project the data onto the 2 primary principal components
        kmeans.plot_in_2d(X, predColours, K, ['Predicted clusters for leaning right extension lengths', xCopLabel, yCopLabel])
        kmeans.plot_in_2d(X, colours, K, ['Defined clusters for leaning right extension lengths', xCopLabel, yCopLabel])
     
        
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
        
        '''
        Using sklearn SVM.SVC - Support Vector Classifier 
        '''
    
        clf = SVC(probability=True, verbose=True)
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        print(y_pred)
        score = clf.score(X_test, y_test)
        print(score)
        '''
        Using my SVM stuff, this can go die.
        '''
        # What kernel would you like to use? Make sure I'm importing the right kernel file.
#        chosenKernel = linear_kernel
#        svm = MySVM(kernel=chosenKernel, C = 1, power=4, coef=2)
#        
#        svm.fit(X_train, y_train)
#        y_pred = svm.predict(X_test)
#    
#        print ("Kernel: {}, Accuracy: {}".format(chosenKernel, accuracy_score(y_test, y_pred)))
#    
#        colours = ['green' if l == 1 else 'red' for l in y_test]
#
#        # Reduce dimensions and plot the results
#        svm.plot_in_2d(X_test, colours, ['',xCopLabel,yCopLabel])
       



    '''
    Shows each participants difference between extension values in a and b tests
    Would like to refactor graphParticipantsAboveBelow into the Participant object make the loop I mentioned in Helper here.
    '''
#    Helper.graphParticipantsAboveBelow(participants, pCount, trial = 2, labels = [xCopLabel, yCopLabel])
#    Helper.saveFigures(participants, 'x & y over time', xCopLabel, yCopLabel)
#    return

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