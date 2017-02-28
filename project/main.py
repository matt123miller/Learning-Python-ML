# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    tl = xy[:,0]
    tr = xy[:,1]
    bl = xy[:,2]
    br = xy[:,3]
    x = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
    return x
    
def centreOfPressureY(xy):
    tl = xy[:,0]
    tr = xy[:,1]
    bl = xy[:,2]
    br = xy[:,3]
    y = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
    return y
    
def extractCopFrom(rawData):

    copX = centreOfPressureX(rawData)
    copY = centreOfPressureY(rawData)
    
    copX = copX[np.logical_not(np.isnan(copX))]
    copY = copY[np.logical_not(np.isnan(copY))]
      
    copX = copX[np.logical_not(np.isinf(copX))]
    copY = copY[np.logical_not(np.isinf(copY))]
                
    return copX, copY

def plotTimeSeriesFrom(rawData):
    
    tl = np.array(rawData[:,0])
    tr = np.array(rawData[:,1])
    bl = np.array(rawData[:,2])
    br = np.array(rawData[:,3])

    axisX = np.arange(len(rawData))

    plt.plot(axisX, tl)
    plt.plot(axisX, tr)
    plt.plot(axisX, bl)
    plt.plot(axisX, br)

    plt.show()

def plotCopLine(rawData):
    copX, copY = extractCopFrom(rawData)
    
    length = np.arange(len(rawData))
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    plt.plot(length, copX)
#    plt.plot(length, copY)
    plt.plot(copX, copY)
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