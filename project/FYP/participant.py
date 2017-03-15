# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
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
        self.plateaus = []
        self.meanPlateauValue = 0.0
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
        
    

    def averageMagnitudeLookAhead(self, by = 30, varianceThreshold = 0.5):
        
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
        
        self.plateaus = np.array(plateaus)
        return self.plateaus
       
    # Should give 1 value for each plateau area. These values will be part of the ML Model
    def averagePlateauSteps(self, plateaus):
        '''
        array to hold each flat part of the palteaus
        rturn array
        loop through through plateaus array
        if i is 0, then we save the mean of avgFlat, then clear avgFlat and move on
        else we save the length of the point to avgFlat
        
        then average any flat parts and append to return array
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

    def showAvgHighLows(self, plateaus, show = False):
        avgPlateauValues = self.averagePlateauSteps(plateaus)
        
        axisX = np.arange(len(avgPlateauValues))
        self.meanPlateauValue = np.mean(avgPlateauValues)
        
        plt.title(self.name)
        plt.scatter(axisX, avgPlateauValues)
        if show:
            plt.show()
            
        return self.meanPlateauValue
            
        
    
    def plotCopLine(self, show = True):
        
        print(len(self.copX))
        print(len(self.copY))
        #length = np.arange(len(rawData))
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection = '3d')
    #    plt.plot(length, copX)
    #    plt.plot(length, copY)
        plt.plot(self.copX, self.copY)
        plt.title(self.name)
        if show:
            plt.show()
        
         
        ## Works great!
    def lineTimeSeriesFrom(self, modifiedData = [], show = True):
       
        data = []
        if len(modifiedData) == 0:
            data = self.data6
        else:
            data = modifiedData
            
        tl = np.array(data[:,0])
        tr = np.array(data[:,1])
        bl = np.array(data[:,2])
        br = np.array(data[:,3])
    
        axisX = np.arange(len(data))
        plt.xlim([-50, len(data) + 50])
    
        plt.title(self.name)
        
        plt.plot(axisX, tl, color = 'b')
        plt.plot(axisX, tr, color = 'c')
        plt.plot(axisX, bl, color = 'r')
        plt.plot(axisX, br, color = 'y')
    
        if show:
            plt.show()
        
    def scatterTimeSeriesFrom(self, modifiedData = [], show = True):
        
        data = []
        if len(modifiedData) == 0:
            data = self.data6
        else:
            data = modifiedData
            
        tl = np.array(data[:,0])
        tr = np.array(data[:,1])
        bl = np.array(data[:,2])
        br = np.array(data[:,3])
    
        axisX = np.arange(len(data))
        plt.xlim([-50, len(data) + 50])
    
        plt.title(self.name)
    #    
        plt.scatter(axisX, tl, color = 'b')
        plt.scatter(axisX, tr, color = 'c')
        plt.scatter(axisX, bl, color = 'r')
        plt.scatter(axisX, br, color = 'y')
        
        if show:
            plt.show()
           
    
    def compoundScatterLine(self, plateaus = []):
        self.scatterTimeSeriesFrom(modifiedData = self.data6[plateaus], show = False)
        self.lineTimeSeriesFrom(show = False)
        plt.show()
  