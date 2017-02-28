# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import csv
import scipy.io
import h5py 

#import hdf5

"""
variables
"""
plateWidth = 100
plateLength = 100

"""
TL = front left
TR = front right
BL = back left
BR = back right
"""

def loadMatlabFile(filename, singlekey, printfile = False):
    matlabFile = scipy.io.loadmat(filename)
    if printfile:
        print(matlabFile)
    return np.array(matlabFile[singlekey])

def centreOfPressureX(xy):
    tl = xy[0]
    tr = xy[1]
    bl = xy[2]
    br = xy[3]
    x = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
    return x
    
def centreOfPressureY(xy):
    tl = xy[0]
    tr = xy[1]
    bl = xy[2]
    br = xy[3]
    y = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
    return y
    
def extractCopFrom(rawData):
    
    copX = np.array([]).astype(float)
    copY = np.array([]).astype(float)
    
    for data in rawData:
        copX = np.append(copX, centreOfPressureX(data))
        copY = np.append(copY, centreOfPressureY(data))
    
    return copX, copY

def plotTimeSeriesFrom(rawData):
    tl = np.array([])
    tr = np.array([])
    bl = np.array([])
    br = np.array([])

    for data in rawData:
        tl = np.append(tl, data[0])
        tr = np.append(tr, data[1])
        bl = np.append(bl, data[2])
        br = np.append(br, data[3])

    axisX = np.arange(len(rawData))

    plt.plot(axisX, tl)
    plt.plot(axisX, tr)
    plt.plot(axisX, bl)
    plt.plot(axisX, br)

    plt.show()

def plotCopLine(rawData):
    copX, copY = extractCopFrom(rawData)
    
    length = np.arange(len(rawData))
    plt.plot(length, copX)
    plt.plot(length, copY)
    plt.show()

    print(copX[2500])

rawData = loadMatlabFile(filename = '0708 Trial3 te.mat', singlekey = 'data6')


# Plot the whole file to see the differences
#plotTimeSeriesFrom(rawData)

# Trying to plot a line graph showing the movement taken
plotCopLine(rawData)

# Testing random new shit
#
#print(rawData[4000])
#x, y = extractCopFrom(rawData)
#print(np.shape(x))
#print(np.shape(y))
# Graph 