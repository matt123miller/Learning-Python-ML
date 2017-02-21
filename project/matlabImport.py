# -*- coding: utf-8 -*-
""" 
scipy.io.loadmat notes:
v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
You will need an HDF5 python library to read matlab 7.3 format mat files. 
Because scipy does not supply one, we do not implement the HDF5 / 7.3 interface here.

h5py and hdf5 are both related to importing and reading matlab files

This file isn't actually that useful in the end as I'm interestd in the 'data6' key in the dictionary
I'm currently working on how to efficiently manipulate that key, which contains roughly 10,000 rows.

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
    

def loadMatlabFile(filename, singlekey):
    matlab = io.loadmat(filename)
    
    dataRaw = np.array(matlab[singlekey])
    
    dataX = np.array([])
    dataY = np.array([])
    
    # TODO Work on getting the right x y data out for graphing
    
    return dataX, dataY
    

#dataX, dataY = loadMatlabFile(filename = '0708 Trial1 TE.mat')
dataX, dataY = loadMatlabFile(filename = '0708 Trial1 TE.mat', singlekey = 'data6')

plt.plot(dataX, dataY) # Give it the x and y axis values
plt.axis([0,10,0,10]) # What is the range of each axis?
plt.xlabel("Pressure on the X axis")
plt.ylabel("Pressure on the Y axis")
plt.title("Centre of pressure")

plt.show()


