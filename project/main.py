# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import csv
import scipy.io as io
import h5py 

#import hdf5


mat = io.loadmat('0708 Trial1 TE.mat')

data = np.array(mat['data6']).astype(float)



csvFile = open('Participant20.csv', newline='\n')

file = csv.reader(csvFile)

rows = {} 
i = 0

for row in file:
    rowdata = np.array(row[1:]).astype(float)
    rows[row[0]] = rowdata
    i += 1 # Couldn't work out how to range with the file       
             
print(rows.keys())