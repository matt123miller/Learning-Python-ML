# -*- coding: utf-8 -*-

import csv
import numpy as np
import scipy.io as io

class Participant:
    
    def __init__(self, name = "", fileType = ".mat"):
        self.name = name
        self.dataBlob = np.array([])
        self.filename = name + fileType
        
        if fileType == ".mat":
            matlab = io.loadmat(self.filename)
            # atm I use data6 as that's what's in the files I was given
            dataBlob = np.array(matlab["data6"]) 
        elif fileType == ".csv":
            csvFile = open(self.filename, newline='\n')
            file = csv.reader(csvFile)
            dataBlob = file["whatever the hell goes here for the magic"]



