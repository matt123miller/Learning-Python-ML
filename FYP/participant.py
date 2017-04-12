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
    participantName and filetype aren't case sensitive
    '''
    def __init__(self, date, participantName = '', trialName = '', fileType = ".mat", dataKey = 'data6', createCSV = False):
        self.participantName = participantName
        self.trialName = trialName
        self.dataBlob = None
        self.copX = np.array([]).astype(float)
        self.copY = np.array([]).astype(float)
        self.copPoints = np.array([]).astype(Point) # will hold points
        self.data6 = np.array([[]]).astype(float)
        self.plateaus = []
        self.avgPlateaus = np.array([]).astype(Point) # The average Points for each plateau section, later split into rest and extension
        self.plateauTargets = [] # Will hold ints for the whole dataset, -1 for resting, 1 for extension
        self.plateauSensorValues = []
        self.meanPoint = Point()
        self.extensionPoints = np.array([])
        self.restPoints = np.array([])
        self.meanRestPoint = Point(0,0)
        self.vectorsBetween = np.array([])
        self.anglesBetween = np.array([]).astype(float)
        self.beginIndex = 0
        self.endIndex = 0
        self.movement = ''
        
        self.filename = date + ' ' + trialName + ' ' + participantName + fileType
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
        
        self.copPoints = [Point(x = self.copX[i], y = self.copY[i]) for i in range(len(self.copX))]


        if createCSV:
            self.createCSV()
        
    
    def generateFeatures(self, byValue, threshold):
        # Returns numpy arrays where possible
        
        plateaus = self.lookAheadForPlateau(by = byValue, varianceThreshold = threshold)
        
        self.avgPlateaus = self.averagePlateauSections(plateaus, 'p')
        
        self.plateauSensorValues = self.extractData6Values(plateaus)
        
        self.extensionPoints, self.restPoints, self.meanPoint = self.splitDataAboveBelowMean(self.avgPlateaus, n_tests = 10, returnType = 'p', cullValues = True) 
        
        # Make my above and below arrays each 10 values long for the 10 tests, hopefully
#        print('There are {} extension point values and {} rest point values'.format(len(self.extensionPoints), len(self.restPoints)))
        
#        self.formatAboveBelowIntoNEach(self.avgPlateaus, n_tests = 10)
        
        self.meanRestPoint = Point.averagePoints(self.restPoints)
        
        '''
        Now that I've got a somewhat normalised value for each plateau above the 
        mean rest point I can graph each participant for their differences between 
        tests a and b for each direction. Then SVM that to get an actual project?
        '''

        self.vectorsBetween = [self.extensionPoints[i] - self.restPoints[i] for i in range(len(self.restPoints))]
        self.anglesBetween = [Point.angleBetween(self.restPoints[i], self.extensionPoints[i]) for i in range(len(self.restPoints))]

                              
    def namesAndFeatures(self):
           return {'plateauSensorValues': self.plateauSensorValues,
                   'restPointsX': [cp.x for cp in self.restPoints],
                   'restPointsY': [cp.y for cp in self.restPoints],
                   'extensionPointsX': [cp.x for cp in self.extensionPoints],
                   'extensionPointsY': [cp.y for cp in self.extensionPoints],
                   'meanPoint': self.meanPoint,
                   'meanRestPoint': self.meanRestPoint,
                   'vectorsBetweenX': [cp.x for cp in self.vectorsBetween],      
                   'vectorsBetweenY': [cp.y for cp in self.vectorsBetween],
                   'anglesBetween': self.anglesBetween
                   }
                              
           
   
    def extractData6Values(self, plateaus):
        returnList = []
        platList = []
        for i, arr in enumerate(plateaus):
            if arr != 0:
                platList.append(self.data6[i])
            elif len(platList) > 0:
                # List comp is used because it will copy by value a new array,
                # but just assigning platList was appending a reference to a list that's then cleared so returnList was empty.
                returnList.append([[plat[0],plat[1],plat[2],plat[3]] for plat in platList])
                platList.clear()
        
        return returnList

        
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
                # We've reached the end, maybe just return?
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
        return array
        loop through through plateaus array
        if i is 0, 
            then we save the mean of avgFlat, 
            then clear avgFlat and move on
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


    def splitDataAboveBelowMean(self, npIn, n_tests, returnType, cullValues = False):
        above = np.array([])
        below = np.array([])
        mean = 0
        
        if returnType == 'm':
            mean = np.mean(npIn)
            above = npIn[npIn > mean]
            below = npIn[npIn < mean]
        else: #it's for points
            mean = Point.averagePoints(npIn).sqrMagnitude()
            
            above = [p for p in npIn if p.sqrMagnitude() > mean]
            below = [p for p in npIn if p.sqrMagnitude() < mean]
        
        if cullValues:
            above, below = above[:n_tests], below[:n_tests]

        return above, below, Point.averagePoints(np.append(above,below))
        
#    def formatAboveBelowIntoNEach(self, avgPlateaus, n_tests):
#        '''
#        loop through the array
#            is this value the same side of the meanpoint as the prevous valiue?
#                add it to an array
#                average that array
#                add it to a return list
#        '''
#        returnAbove, returnBelow = [], []
#        avgList = []
#        midMag = self.meanPoint.magnitude()     
#        isAbove = avgPlateaus[0].magnitude() > midMag
#
#        prevMag = 0.0
#        
#        def avg(inList):
#            return 0
#        
#        for i, point in enumerate(avgPlateaus):
#            if i == 0: # Skip the first iteration
#                prevMag = point.magnitude()
#                continue
#            greaterThan = point.magnitude() > midMag:
#            
#            if isAbove and greaterThan:
#                avgList.append(point)
#            elif isAbove and not greaterThan:
#                returnAbove.append(point)
#            elif not isAbove and greaterThan:
#                
#            elif not isAbove and not greaterThan:
#                pass
#        return 0
#    
#
#    def fillPlateauTargets(self, aboveTarget, belowTarget):
#        
#        return 0
    
       
    def plotAvgHighLows(self, avgPlateauValues, show = False):
                
        axisX = np.arange(len(avgPlateauValues))
        
        plt.title(self.name)
        plt.scatter(axisX, avgPlateauValues)
        if show:
            plt.show()
 
    def plotCopHighLows(self):
        plt.scatter([c.x for c in self.extensionPoints], [c.y for c in self.extensionPoints], color = 'r')
        plt.scatter([c.x for c in self.restPoints], [c.y for c in self.restPoints], color = 'g')
        plt.title(self.name)
        plt.show()
    
    def plotCopLine(self, show = True):
        
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
   
    
    '''
    Boring methods
    '''
    
    def createCSV(self):
        '''
        Make a CSV file out of the COP data or maybe the data6
        Create a csv writer
        write the data
        save the file
        'w' parameter will write and overwrite if it exists already 
        '''
        with open('{}{}'.format(self.name, '.csv'), 'w') as file:
            writer = csv.writer(file, dialect='excel')
            for cp in self.copPoints:
                writer.writerow([cp.printForUnity()])
                
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
        