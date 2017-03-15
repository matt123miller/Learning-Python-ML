# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from participant import Participant
from point import Point


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
    # Do I want a specific subset of files for some reason?
#    participantNames = ['mm'] # e.g. just my data
    # pass this to all methods
    participants = loadParticipants(trials = trialNames, names = participantNames)
    
    for p in participants[4:5]:
        data6 = p.data6
        byValue = 15
        threshold = 0.5
        plateaus = p.averageMagnitudeLookAhead(by = byValue, varianceThreshold = threshold)
        print('The plateaus were computed by looking {} values ahead and saving values below {}'.format(byValue, threshold))
        p.showAvgHighLows(plateaus, show = True)
        p.compoundScatterLine(plateaus)
        
    
    

if __name__ == "__main__":
    main()