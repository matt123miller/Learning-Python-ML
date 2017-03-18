# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from participant import Participant
from point import Point
from HelperFile import Helper


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



def loadParticipants(trials, names):
    outParticipants = []

    for trial in trials:
        for name in names:
            p = Participant(name = date + ' ' + trial + ' ' + name, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants

    
def main():
        
    pCount = len(participantNames)
    tCount = len(trialNames)
    byValue = 15
    threshold = 0.5
    returnType = 'p'
    # Do I want a specific subset of files for some reason?
#    participantNames = ['mm'] # e.g. just my data
#    trialNames = [trialNames[:1]]
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
    Introduce a loop here that will create graphs and whatnot out of each individual Participant
    Much cleaner
      
        for i in whatever:
            do things
    '''  
    
    print('All data manipulation is hopefully done now \nNow to make graphs and things out of each participant')
    
    # 0 for trial 1a and 1b, 2 for 2a and 2b, 4 for 3a and 3b
    rfrom = pCount * 4 
    rto = rfrom + pCount
    
    for i in range(rfrom, rto):
        testA = participants[i]
        testB = participants[i + pCount] # This will fetch the same participants alternate movement test.
        
        ''' Data '''
        aX = [cp.x for cp in testA.normalisedAboveMean]
        aY = [cp.y for cp in testA.normalisedAboveMean]
        bX = [cp.x for cp in testB.normalisedAboveMean]
        bY = [cp.y for cp in testB.normalisedAboveMean]
        
        ''' Labels and things '''
        testAName = testA.name.split()[1]
        testBName = testB.name.split()[1]
        pName = testA.name.split()[2].upper()
        title = 'CoP values during extension for\n{} (green) vs {} (red) for participant {}'.format(testAName, testBName, pName)
        
        ''' Graph all the thing '''
        plt.scatter(aX, aY, color = 'g')
        plt.scatter(bX, bY, color = 'r')
#        plt.xlim([-1,3])a
#        plt.ylim([-6,6])
        plt.xlabel(xCopLabel)
        plt.ylabel(yCopLabel)
        plt.title(title)
        plt.show()


    '''
    Finally a section for making any graphics out of ALL data
    '''
#    if returnType == 'p':
#        # make a graph of the points, cartesian style
#
#        xDataLow = [cp.x for cp in lowMeans]
#        xDataHigh = [cp.x for cp in highMeans]
#        yDataLow = [cp.y for cp in lowMeans]
#        yDataHigh = [cp.y for cp in highMeans]
#        plt.scatter(xDataLow, yDataLow, color = 'g')
#        plt.scatter(xDataHigh, yDataHigh, color = 'r')
#        plt.title('Shows the average CoP values when at rest (in green)\nand extension (in red) for all participants during test 1a')
#        plt.xlabel('CoP values on the x axis')
#        plt.ylabel('CoP values on the y axis')
#        plt.show()
#        
#    elif returnType == 'm':
#        
#        print('{}'.format(highMeans))
#        print('{}'.format(lowMeans))
#        diffMeans = highMeans - lowMeans
#        print(diffMeans)
        
#    axis = np.arange(len(meanDiffs))
#    plt.scatter(axis, highMeans, color= 'r')
#    plt.scatter(highMeans, lowMeans, color = 'g')
#    plt.show()


if __name__ == "__main__":
    main()