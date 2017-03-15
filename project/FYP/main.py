# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from participant import Participant
from point import Point
from HelperFile import Helper

#import hdf5

"""
variables
"""
plateWidth = 100
plateLength = 100
participantNames = ['ac', 'an', 'gd', 'gp', 'hm', 'lh', 'mb', 'mm', 'te', 'wy', 'xd', 'yz'] # Not case sensitive
trialNames = ['Trial1a','Trial1b','Trial2a','Trial2b','Trial3a','Trial3b'] # not case sensitive
date = "170306"
dataKey = 'data6'


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

    # pass this to all methods
    # use a slice on trialnames or participant names!
    participants = loadParticipants(trials = trialNames, names = participantNames)
    highMeans = np.array([])
    lowMeans = np.array([])
    diffMeans = np.array([])
    
    print('The plateaus were computed by looking {0} values ahead and saving values below {1}'.format(byValue, threshold))

    for p in participants:
        
        plateaus = p.lookAheadForPlateau(by = byValue, varianceThreshold = threshold)
        
        avgPlateaus = p.averagePlateauSections(plateaus, returnType)
        #returns numpy arrays
        highs, lows = Helper.splitDataAboveBelowMean(avgPlateaus, returnType) 
        
        if returnType == 'p':
            plt.scatter([p.x for p in highs], [p.y for p in highs], color = 'r')
            plt.scatter([p.x for p in lows], [p.y for p in lows], color = 'g')
            plt.title(p.name)
            plt.show()
        
            highMeans = np.append(highMeans, Helper.averagePoints(highs))
            lowMeans = np.append(lowMeans, Helper.averagePoints(lows))
            
        else:
            highMean = np.mean(highs)
            lowMean = np.mean(lows)
            print('The highs are {} with a mean of {}'.format(highs, highMean))
            print('the lows are{} with a mean of {}'.format(lows, lowMean))
            print('') #empty row
            highMeans = np.append(highMeans, highMean)
            lowMeans = np.append(lowMeans, lowMean)
        
#        p.compoundScatterLine(plateaus)
#        p.showAvgHighLows(avgPlateaus, show = True)
#        plt.scatter(np.arange(len(avgPlateaus)), avgPlateaus)
        
    print('{}'.format(highMeans))
    print('{}'.format(lowMeans))
    diffMeans = highMeans - lowMeans
    print(diffMeans)
#    axis = np.arange(len(meanDiffs))
#    plt.scatter(axis, highMeans, color= 'r')
#    plt.scatter(highMeans, lowMeans, color = 'g')
#    plt.show()


if __name__ == "__main__":
    main()