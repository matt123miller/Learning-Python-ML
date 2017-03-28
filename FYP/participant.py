# -*- coding: utf-8 -*-

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import math

from point import Point
from HelperFile import Helper

#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, dir_path + "/Data")
#from Data import *

# Just learnt you have to still use self to make sure you're setting
# The version of the variable that's public to the creator of this object
class Participant(object):
    
    '''
    name and filetype aren't case sensitive
    '''
    def __init__(self, name = "", fileType = ".mat", dataKey = 'data6', createCSV = False):
        self.name = name
        self.dataBlob = None
        self.copX = np.array([]).astype(float)
        self.copY = np.array([]).astype(float)
        self.copPoints = np.array([]).astype(Point) # will hold points
        self.data6 = np.array([[]]).astype(float)
        self.plateaus = []
        self.meanAllPlateaus = 0.0
        self.aboveMean = np.array([])
        self.belowMean = np.array([])
        self.meanRestPoint = Point(0,0)
        self.extensionDifferences = np.array([])
        self.beginIndex = 0
        self.endIndex = 0
        self.movement = ''
        
        self.filename = name + fileType
        self.dataKey = dataKey
        
        if fileType == ".mat":
            matlab = io.loadmat(self.filename)
            # atm I use data6 as that's what's in the files I was given
            self.dataBlob = matlab
        '''
        Not gonna use csv for now, too much extra complexity.
        elif fileType == ".csv":
            csvFile = open(self.filename, newline='\n')
            file = csv.reader(csvFile)
            self.dataBlob = file["whatever the hell goes here for the magic"]
        '''
        self.data6 = self.dataBlob[dataKey]
        
        self.stripOutEnds(minimumSensorThreshold = 400)
        self.removeJunkData()    
        
        for i in range(len(self.copX)):
            self.copPoints = np.append(self.copPoints, Point(x = self.copX[i], y = self.copY[i]))
         
        if createCSV:
            self.createCSV()
        
         
    
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
       
        
    def stripOutEnds(self, minimumSensorThreshold):
                
        # I could try some list comprehension magic but I'd rather keep it clear to my noob python brain
        # And I also need some numbers saved along with the lists themselves
        beginFound = False
        endFound = False
        # loop through the front
        for i in range(len(self.data6)):
            data = self.data6[i,:]
            
            for d in data:
                if d > minimumSensorThreshold:
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
                if d > minimumSensorThreshold:
                    self.endIndex = j
                    endFound = True
                    break
            
            if endFound:
                break
        
        self.data6 = self.data6[self.beginIndex-1:self.endIndex+1]
        self.copX = self.dataBlob['result_x'][self.beginIndex-1:self.endIndex+1]
        self.copY = self.dataBlob['result_y'][self.beginIndex-1:self.endIndex+1]
        
    
    def createCSV(self, overwrite = False):
        '''
        Make a CSV file out of the COP data or maybe the data6
        Create a csv writer
        write the data
        save the file
        '''
        writeData = []
        
        ''' 'w' parameter will write and overwrite if it exists already '''
        with open('{}{}'.format(self.name, '.csv'), 'w') as file:
            writer = csv.writer(file, dialect='excel')
            for p in self.copPoints:
                writer.writerow([p.printForUnity()])
#            
   
        
    
    def normaliseData(self):
        self.copX = Point.normaliseOverHighestValue(self.copX)
        self.copY = Point.normaliseOverHighestValue(self.copY)
        self.copPoints = [p.normalise() for p in self.copPoints]
    '''
    Returns an array of length data6.count containing zeroes or an index where a flat point is.
    '''
    def lookAheadForPlateau(self, by = 30, varianceThreshold = 0.5):
        
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
       
    '''
    Should give 1 value for each plateau area. These values will be part of the ML Model
    returnTypes: m is magnitudes, p is points
    '''
    def averagePlateauSections(self, plateaus, returnType = 'm'):
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
                if len(avgFlat) != 0: #we've reached the end of a plateau
                    if returnType == 'm':
                        returnArray.append(np.mean(avgFlat))
                    else:
                        returnArray.append(Point.averagePoints(avgFlat))
                    avgFlat.clear()
                continue
            if returnType == 'm': 
                avgFlat.append(self.copPoints[i].magnitude())
            else:
                avgFlat.append(self.copPoints[i])

        # Consider the code reaching the end of the list with items not yet averaged
        if len(avgFlat) != 0:
            if returnType == 'm':
                returnArray.append(np.mean(avgFlat))
            else:
                returnArray.append(Point.averagePoints(avgFlat))

        
        return np.array(returnArray)

    def showAvgHighLows(self, avgPlateauValues, show = False):
                
        axisX = np.arange(len(avgPlateauValues))
        
        plt.title(self.name)
        plt.scatter(axisX, avgPlateauValues)
        if show:
            plt.show()
 
    def plotCopHighLows(self):
        plt.scatter([c.x for c in self.aboveMean], [c.y for c in self.aboveMean], color = 'r')
        plt.scatter([c.x for c in self.belowMean], [c.y for c in self.belowMean], color = 'g')
        plt.title(self.name)
        plt.show()
    
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
  