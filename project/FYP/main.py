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


#Works less great
def plotCopLine(copX, copY, title = ''):
    
    print(len(copX))
    print(len(copY))
    #length = np.arange(len(rawData))
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    plt.plot(length, copX)
#    plt.plot(length, copY)
    plt.plot(copX, copY)
    plt.title(title)
    plt.show()
    
    
## Works great!
def plotTimeSeriesFrom(rawData, title = ''):
    
    tl = np.array(rawData[:,0])
    tr = np.array(rawData[:,1])
    bl = np.array(rawData[:,2])
    br = np.array(rawData[:,3])

    axisX = np.arange(len(tl))

    plt.title(title)
    plt.plot(axisX, tl, color = 'b')
    plt.plot(axisX, tr, color = 'c')
    plt.plot(axisX, bl, color = 'r')
    plt.plot(axisX, br, color = 'y')

    plt.show()



def loadParticipants(trials, names):
    outParticipants = []

    for trial in trials:
        for name in names:
            p = Participant(name = date + ' ' + trial + ' ' + name, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants
  
    
def main():
        
    # Do I want a specific subset of files for some reason?
#    participantNames = ['mm'] # e.g. just my data
    # pass this to all methods
    participants = loadParticipants(trials = trialNames, names = participantNames)
    
    for p in participants[:6]:
        data6 = p.data6
        plateaus = p.averageMagnitudeLookAhead(by = 10)
        print(plateaus)
        print(p.copPoints[230])
#        plotCopLine(p.copX, p.copY, title = p.name) 
#        plotTimeSeriesFrom(data6, title = p.name)
    

    

if __name__ == "__main__":
    main()