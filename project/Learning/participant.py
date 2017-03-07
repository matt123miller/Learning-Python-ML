# -*- coding: utf-8 -*-

import csv
import numpy as np
import scipy.io as io

# Just learnt you have to still use self to make sure you're setting
# The version of the variable that's public to the creator of this object
class Participant:
    
    def __init__(self, name = "", fileType = ".mat"):
        self.name = name
        self.dataBlob = None
        self.filename = name + fileType
        
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
        
        for 
        return None
        
    def runningAvgAlgo(avgThreshold = 5):
        
        
        return None
        


    