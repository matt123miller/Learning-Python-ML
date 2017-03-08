# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from participant import Participant

#import hdf5

"""
variables
"""
plateWidth = 100
plateLength = 100
participantNames = ['wy','xd','te'] # Not case sensitive
trialNames = ['Trial1','Trial2','Trial3'] # not case sensitive
date = "0708"
dataKey = 'data6'


"""
TL = front left = data6[[:,0]]
TR = front right = data6[[:,1]]
BL = back left = data6[[:,2]]
BR = back right = data6[[:,3]]
"""

"""
Trying to move to the participant, it seems appropriate
#def centreOfPressureX(xy):
#    tl = xy[:,0]
#    tr = xy[:,1]
#    bl = xy[:,2]
#    br = xy[:,3]
#    x = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
#    return x
#    
#def centreOfPressureY(xy):
#    tl = xy[:,0]
#    tr = xy[:,1]
#    bl = xy[:,2]
#    br = xy[:,3]
#    y = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
#    return y
    
def extractCopFrom(rawData):

    copX = centreOfPressureX(rawData)
    copY = centreOfPressureY(rawData)
    
    copX = copX[np.logical_not(np.isnan(copX))]
    copY = copY[np.logical_not(np.isnan(copY))]
      
    copX = copX[np.logical_not(np.isinf(copX))]
    copY = copY[np.logical_not(np.isinf(copY))]
                
    return copX, copY
"""

#Works less great
def plotCopLine(rawData):
    copX, copY = extractCopFrom(rawData)
    
    length = np.arange(len(rawData))
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    plt.plot(length, copX)
#    plt.plot(length, copY)
    plt.plot(copX, copY)
    plt.label
    plt.show()
    
    print(length)
    print(len(copX))
    print(len(copY))
    
## Works great!
def plotTimeSeriesFrom(rawData, title = ''):
    
    tl = np.array(rawData[2000:4000,0])
    tr = np.array(rawData[2000:4000,1])
    bl = np.array(rawData[2000:4000,2])
    br = np.array(rawData[2000:4000,3])

    axisX = np.arange(len(tl))

    plt.title(title)
    plt.plot(axisX, tl, color = 'b')
    plt.plot(axisX, tr, color = 'c')
    plt.plot(axisX, bl, color = 'r')
    plt.plot(axisX, br, color = 'y')

    plt.show()



def loadParticipants():
    outParticipants = []

    for name in participantNames:
        for trial in trialNames:
            p = Participant(name = date + ' ' + trial + ' ' + name, fileType = '.mat')
            outParticipants.append(p)
    
    return outParticipants
  
    
def main():
        
    # pass this to all methods
    participants = loadParticipants()
    
    for p in participants:
        data6 = p.dataBlob[dataKey]
        plotTimeSeriesFrom(data6, title = p.name)
#    
#    rawData = p.dataBlob[dataKey]
#    print(rawData)
#    # Plot the whole file to see the differences
#    plotTimeSeriesFrom(rawData)
    
    # Trying to plot a line graph showing the movement taken
    #plotCopLine(rawData)
    
    # Testing random new shit
    
    #
    #print(rawData[4000])
    #x, y = extractCopFrom(rawData)
    #print(np.shape(x))
    #print(np.shape(y))
    # Graph 
    

if __name__ == "__main__":
    main()