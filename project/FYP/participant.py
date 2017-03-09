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
        self.copX = np.array([]).astype(float)
        self.copY = np.array([]).astype(float)
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
        

    def runningAvgAlgo(self,avgThreshold = 5):
        
        
        return None
        
    
#    def centreOfPressureX(xy, plateLength = 100):
#        tl = xy[:,0]
#        tr = xy[:,1]
#        bl = xy[:,2]
#        br = xy[:,3]
#        x = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateLength * 0.5)
#        return x
#    
#    def centreOfPressureY(xy, plateWidth = 100):
#        tl = xy[:,0]
#        tr = xy[:,1]
#        bl = xy[:,2]
#        br = xy[:,3]
#        y = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)
#        return y
#        
#    
#    def extractCopFrom(data, plateWidth = 100, plateHeight = 100):
#
#        tl = data[:,0]
#        tr = data[:,1]
#        bl = data[:,2]
#        br = data[:,3]
#        copX = ((tr + br - tl - bl)/(tr+br+tl+bl)) * (plateWidth * 0.5)
#        copY = ((tl + tr - bl - br)/(tr+br+tl+bl)) * (plateWidth * 0.5)

        # Remove NaN's and infinites
        ## Used to be 2 separate operations but I pass 2 booleans with and instead now, Hopefully it works.
#        copX = copX[np.logical_not(np.isnan(copX)) and np.logical_not(np.isinf(copX))]
#        copY = copY[np.logical_not(np.isnan(copY)) and np.logical_not(np.isinf(copY))]
##        copX = copX[]
##        copY = copY[]
#            
#        return copX, copY   
    
