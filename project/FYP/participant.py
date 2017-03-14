# -*- coding: utf-8 -*-

import csv
import numpy as np
import scipy.io as io
import math

from point import Point

# Just learnt you have to still use self to make sure you're setting
# The version of the variable that's public to the creator of this object
class Participant:
    
    '''
    name and filetype aren't case sensitive
    '''
    def __init__(self, name = "", fileType = ".mat", dataKey = 'data6'):
        self.name = name
        self.dataBlob = None
        self.copX = np.array([]).astype(float)
        self.copY = np.array([]).astype(float)
        self.copPoints = np.array([]).astype(Point) # will hold points
        self.data6 = np.array([[]]).astype(float)
        self.beginIndex = 0
        self.endIndex = 0
        
        self.filename = name + fileType
        self.dataKey = dataKey
        
        if fileType == ".mat":
            matlab = io.loadmat(self.filename)
            # atm I use data6 as that's what's in the files I was given
            self.dataBlob = matlab
        # Not gonna use csv for now, too much extra complexity.
#        elif fileType == ".csv":
#            csvFile = open(self.filename, newline='\n')
#            file = csv.reader(csvFile)
#            self.dataBlob = file["whatever the hell goes here for the magic"]
#        
        self.data6 = self.dataBlob[dataKey]
        
        self.stripOutEnds(minimumThreshold = 400)
        self.removeJunkData()    
        
        for i in range(len(self.copX)):
            self.copPoints = np.append(self.copPoints, Point(x = self.copX[i], y = self.copY[i]))
         
    
    def removeJunkData(self):
        
        tempDict = {'copX' : [], 'copY' : [], 'data6' : []}
        
        for i in range(len(self.data6)):
            dataX = self.copX[i]
            dataY = self.copY[i]
            dataSix = self.data6[i]
            
            xisgood = np.logical_not(np.isnan(dataX)) or np.logical_not(np.isinf(dataX))
            yisgood = np.logical_not(np.isnan(dataY)) or np.logical_not(np.isinf(dataY))
            
            if xisgood and yisgood:
                tempDict['copX'].append(dataX)
                tempDict['copY'].append(dataY)
                tempDict['data6'].append(dataSix)

               
        self.copX = np.array(tempDict['copX'])
        self.copY = np.array(tempDict['copY'])
        self.data6 = np.array(tempDict['data6'])
       
        
    def stripOutEnds(self, minimumThreshold):
                
        # I could try some list comprehension magic but I'd rather keep it clear to my noob python brain
        # And I also need some numbers saved along with the lists themselves
        beginFound = False
        endFound = False
        # loop through the front
        for i in range(len(self.data6)):
            data = self.data6[i,:]
            
            for d in data:
                if d > minimumThreshold:
                    self.beginIndex = i
                    beginFound = True
                    break
            
            if beginFound:
                break
        
#        # Loop through the back
        for j in reversed(range(len(self.data6))):
            data = self.data6[j,:]
#            print(j)
            for d in data:
                if d > minimumThreshold:
                    self.endIndex = j
                    endFound = True
                    break
            
            if endFound:
                break
        
        self.data6 = self.data6[self.beginIndex-1:self.endIndex+1]
        self.copX = self.dataBlob['result_x'][self.beginIndex-1:self.endIndex+1]
        self.copY = self.dataBlob['result_y'][self.beginIndex-1:self.endIndex+1]
        
    

    def averageMagnitudeLookAhead(self, by = 5, varianceThreshold = 1):
        
        length = len(self.copPoints)
        plateaus = []
        sqrVar = varianceThreshold ** 2
        
        for i in range(length):
            nextIndex = i + by
            if nextIndex >= length:
                # We've reached the end, maybe just return
                continue
            
            nextItem = self.copPoints[nextIndex]
            diff = (nextItem - self.copPoints[i]).sqrMagnitude()
            
            if diff < sqrVar and diff > -sqrVar:
                plateaus.append(i)
            else:
                plateaus.append(0)
        
        return np.array(plateaus)
       
    # Should give 1 value for each plateau area. These values will be part of the ML Model
    def averagePlateauSteps(self, plateaus):
        '''
        array to hold each flat part of the palteaus
        rturn array
        loop through range of plateaus array
        append to flat parts array UNLESS a 0 is found:
            then average the flat parts array and append to return array
        return array
        '''
        avgFlat, returnArray = [],[]
        for i in plateaus:
            if i == 0:
                if len(avgFlat) != 0:
                    returnArray.append(np.mean(avgFlat))
                    avgFlat.clear()
                continue
            avgFlat.append(self.copPoints[i].magnitude())
        # Consider the code reaching the end of the list with items not yet averaged
        if len(avgFlat) != 0:
            returnArray.append(np.mean(avgFlat))
        
        return returnArray
