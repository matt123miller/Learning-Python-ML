# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import csv
import scipy.io as io
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

def loadMatlabFile(filename, singlekey):
    matlab = io.loadmat(filename)
    
    return np.array(matlab[singlekey])

def centreOfPressureX(xy):
    tl = xy[0]
    tr = xy[1]
    bl = xy[2]
    br = xy[3]
    return ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
    
def centreOfPressureY(xy):
    tl = xy[0]
    tr = xy[1]
    bl = xy[2]
    br = xy[3]
    return ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
    
def extractCopFrom(rawData):
    
    copX = np.array([])
    copY = np.array([])
    
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

rawData = loadMatlabFile(filename = '0708 Trial1 TE.mat', singlekey = 'data6')
#copX, copY = extractCopFrom(rawData)
plotTimeSeriesFrom(rawData)


#plt.plot(copY, len(copY))
#plt.show()

print(len(copX))
print(len(copY))