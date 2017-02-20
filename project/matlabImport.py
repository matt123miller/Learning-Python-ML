# -*- coding: utf-8 -*-
""" 
scipy.io.loadmat notes:
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
You will need an HDF5 python library to read matlab 7.3 format mat files. 
Because scipy does not supply one, we do not implement the HDF5 / 7.3 interface here.

h5py and hdf5 are both related to importing and reading matlab files
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.io as io
import h5py 

def loadMatlabFile(filename):
    matlab = io.loadmat(filename)

#    print(matlab)
    dataXRaw = np.array(matlab['data_x']).astype(float)
    dataYRaw = np.array(matlab['data_y']).astype(float)

    dataX = np.array([])
    dataY = np.array([])
    
    # Loops are required to flatten the raw data arrays
    # You also have to use array1 = numpy.append(array1, array2) to add to an existing array 
    for x in dataXRaw:
        dataX = np.append(dataX, x)
    
    for y in dataYRaw:
        dataY = np.append(dataY, y)
    
    return dataX, dataY
    
dataX, dataY = loadMatlabFile(filename = '0708 Trial1 TE.mat')

plt.plot(dataX, dataY) # Give it the x and y axis values
plt.axis([0,10,0,10]) # What is the range of each axis?
plt.xlabel("Pressure on the X axis")
plt.ylabel("Pressure on the Y axis")
plt.title("Centre of pressure")

plt.show()


