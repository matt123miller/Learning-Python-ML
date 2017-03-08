# -*- coding: utf-8 -*-

import csv
import numpy as np
import scipy.io as io

# Just learnt you have to still use self to make sure you're setting
# The version of the variable that's public to the creator of this object
class Participant:
    '''
    name and filetype aren't case sensitive
    '''
    def __init__(self, name = "", fileType = ".mat", dataKey = 'data6'):
        self.name = name
        self.dataBlob = None
        self.filename = name + fileType
        self.dataKey = dataKey
        
        if fileType == ".mat":
            matlab = io.loadmat(self.filename)
            # atm I use data6 as that's what's in the files I was given
            self.dataBlob = matlab
        elif fileType == ".csv":
            csvFile = open(self.filename, newline='\n')
            file = csv.reader(csvFile)
            self.dataBlob = file["whatever the hell goes here for the magic"]
        
        print("")
    
    def stripOutEnds():
        
        
        return None
        
    def runningAvgAlgo(avgThreshold = 5):
        
        
        return None
    
    def centreOfPressureX(xy, plateLength = 100):
        tl = xy[:,0]
        tr = xy[:,1]
        bl = xy[:,2]
        br = xy[:,3]
        x = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
        return x
    
    def centreOfPressureY(xy, plateWidth = 100):
        tl = xy[:,0]
        tr = xy[:,1]
        bl = xy[:,2]
        br = xy[:,3]
        y = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
        return y
        
    
    def extractCopFrom(data, plateWidth = 100, plateHeight = 100):

        tl = data[:,0]
        tr = data[:,1]
        bl = data[:,2]
        br = data[:,3]
        copX = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateWidth * 0.5)
        copY = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)

        # Remove NaN's and infinites
        ## Used to be 2 separate operations but I pass 2 booleans with and instead now, Hopefully it works.
        copX = copX[np.logical_not(np.isnan(copX)) and np.logical_not(np.isinf(copX))]
        copY = copY[np.logical_not(np.isnan(copY)) and np.logical_not(np.isinf(copY))]
#        copX = copX[]
#        copY = copY[]
            
        return copX, copY   
    
