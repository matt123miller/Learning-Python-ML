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
        self.participantName = participantName # The participants intials
        self.trialName = trialName # The name of the trail set being performed
        self.dataBlob = {} # Contains the whole data object loaded from the file
        self.data6 = np.array([[]]).astype(int) # The main dictionary in the loaded file containing most data
        self.movement = ''
        self.direction = ''
        # Used for finding the beginning and end of meaningful participant data
        self.beginIndex = 0
        self.endIndex = 0
        #### Features ####
        self.copX = np.array([]).astype(float) # The time series X values from the force plate
        self.copY = np.array([]).astype(float) # The time series Y values from the force plate
        self.copPoints = np.array([]).astype(Point) # Cartesian points created from the copX and copY lists
        self.plateaus = [] # Markers for where plateaus in the data are, 1 for a plateau value, otherwise 0
        self.plateauPoints = np.array([]).astype(Point) # The average Points for each plateau section, later split into rest and extension
        self.plateauSensorValues = [] # Contains lists of the 4 sensor values when the data plateaus
        self.plateauSensorAverages = [] # The 4 sensor values averaged for each plateau
        self.restSensors = []
        self.extensionSensors = []
        self.meanPoint = Point() # The average cartesian point of all plateau averages
        self.extensionPoints = np.array([]) # The points from extension plateaus, assumed to be above the meanPoint
        self.restPoints = np.array([]) # The points from rest plateaus, assumed to be below the meanPoint
        self.meanRestPoint = Point(0,0) # The mean of restPoints
        self.vectorsBetween = np.array([]) # The vectors between each restPoint and extensionPoint
        self.anglesBetween = np.array([]).astype(float) # The angles between each restPoint and extensionPoint
        
        
        self.fileName = date + ' ' + trialName + ' ' + participantName
        self.dataKey = dataKey
        
        # Is it a hinge or pendulum trial participant?
        pendulumTrials = ('1a','2a','3a')
        if any(s in trialName.lower() for s in pendulumTrials):
            self.movement = 'pendulum'
        else:
            self.movement = 'hinge'
        
        # What direction are we extending?
        if '1' in trialName:
            self.direction = 'right'
        elif '2' in trialName:
            self.direction = 'forward'
        elif '3' in trialName:
            self.direction = 'backward'
        
        if fileType == ".mat":
            matlab = io.loadmat(self.fileName + fileType)
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
        
        self.plateaus = self.lookAheadForPlateau(by = byValue, varianceThreshold = threshold)
#        print(len([p for p in self.plateaus if p == 1]))
        # Maybe we still need this method, I'll leave this as a reminder for now. 
#        self.plateaus = self.removeSmallPlateaus(5, self.plateaus)
#        print(len([p for p in self.plateaus if p > 0]))

        self.plateauPoints = self.averagePlateauSections(self.plateaus, 'p')

        self.plateauSensorValues = self.extractData6Values(self.plateaus)
        self.plateauSensorAverages = self.avgSensorValues(self.plateauSensorValues)
        
        self.extensionPoints, self.restPoints, self.meanPoint, self.extensionSensors, self.restSensors = self.splitDataAboveBelowMean(self.plateauPoints, self.plateauSensorAverages, returnType = 'p') 
        
        self.meanRestPoint = Point.averagePoints(self.restPoints)
        
        self.trimLists()                  

        self.vectorsBetween = [self.extensionPoints[i] - self.restPoints[i] for i, val in enumerate(self.restPoints)]
        # Does argument order matter for the angles?? nope!!
        self.anglesBetween = [Point.angleBetween(self.extensionPoints[i] ,self.restPoints[i] ) for i, val in enumerate(self.restPoints)]
           

                          
    # I'm too lazy to check if there's a reference ro value issue so I'm just copying everything . Lazy is lazy.
    def listFeaturesDict(self):
           return {'trialType':self.movement,
                   'direction':self.direction,
                   'features':[np.array(self.restSensors),
                               np.array(self.extensionSensors),
                               np.array([cp.x.item() for cp in self.restPoints]),
                               np.array([cp.y.item() for cp in self.restPoints]),
                               np.array([cp.x.item() for cp in self.extensionPoints]),
                               np.array([cp.y.item() for cp in self.extensionPoints]),
                               np.array([cp.x.item() for cp in self.vectorsBetween]),      
                               np.array([cp.y.item() for cp in self.vectorsBetween]),
                               np.array(self.anglesBetween) ],
                   'featureNames':['restSensorValues','extensionSensorValues','restPointsX','restPointsY','extensionPointsX',
                                   'extensionPointsY','vectorsBetweenX', 'vectorsBetweenY', 'anglesBetween'],
                   'participant':self # Just in case we need the whole thing!
                   }
                              
    def singleFeaturesDict(self):
        # Maybe these point values should be represented in a np list fashion [x, y] ?? Might make some things easier later.
        return {'trialType':self.movement,
                'direction':self.direction,
                'features':{'meanPoint': self.meanPoint,
                            'meanRestPoint': self.meanRestPoint
                            },
                'participant':self
                }
    
    def trimLists(self):            
        # Each participants data at this point is a list of a non uniform shape. Make it uniform
        # This will make the length of all features lists be the same as the shortest list.
        # A bunch of min(len()) magic, can you do min with many arguments? Looks like you can.
        # Is there a cool numpy method for this?
        
        restLength, extLength, restSensorLength, extSensorLength = len(self.restPoints), len(self.extensionPoints), len(self.restSensors), len(self.extensionSensors)
        
        # Cull some values so everything is the length of the lowest common denominator
        minLength = min(restLength, extLength, restSensorLength, extSensorLength)
        
        self.restPoints = self.restPoints[:minLength]
        self.extensionPoints = self.extensionPoints[:minLength]
        self.restSensors = self.restSensors[:minLength]
        self.extensionSensors = self.extensionSensors[:minLength]
        return 0
        
    '''
    Returns an array of length data6.count containing zeroes or an index where a flat point is.
    '''
    def lookAheadForPlateau(self, by = 30, varianceThreshold = 0.5):
        
        length = len(self.copPoints)
        plateaus = np.array([0 for _ in range(length)])
        sqrVar = varianceThreshold ** 2
        
        for i in range(length):
            nextIndex = i + by
            if nextIndex >= length:
                # We've reached the end, maybe just return?
                continue
            
            nextItem = self.copPoints[nextIndex]
            diff = (nextItem - self.copPoints[i]).sqrMagnitude()

            if diff < sqrVar and diff > -sqrVar:
                plateaus[i:nextIndex] = 1
        
        return np.array(plateaus)
    
    def removeSmallPlateaus(self, smallLength, plateaus):
        platCounter = 0
        returnPlat = np.array([i for i in plateaus])
        for i, num in enumerate(returnPlat):
            
            if num > 0:
                platCounter += 1
            elif platCounter > 0 and platCounter <= smallLength:
                returnPlat[i - platCounter : i + 1] = 0
                platCounter = 0
            else:
                platCounter = 0
        
        return returnPlat
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
        
        for i, num in enumerate(plateaus):
            if num == 0:
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

        
    def splitDataAboveBelowMean(self, npIn, sensorsIn, returnType, n_tests = -1):
        above = []
        below = []
        restSensors = []
        extSensors = []
        mean = 0
        
        if returnType == 'm':
            mean = np.mean(npIn)
            above = npIn[npIn > mean]
            below = npIn[npIn < mean]
        else: #it's for points
            mean = Point.averagePoints(npIn).sqrMagnitude()
            
            for i, p in enumerate(npIn):
                if p.sqrMagnitude() > mean:
                    above.append(p)
                    extSensors.append(sensorsIn[i])
                else:
                    below.append(p)
                    restSensors.append(sensorsIn[i])
                    
#            above = [p for p in npIn if p.sqrMagnitude() > mean]
#            below = [p for p in npIn if p.sqrMagnitude() < mean]
        
        if n_tests >= 1:
            print("n_tests is used!")
            above, below, restSensors, extSensors = above[:n_tests], below[:n_tests], restSensors[:n_tests], extSensors[:n_tests]

        return np.array(above), np.array(below), Point.averagePoints(np.append(above,below)), np.array(extSensors), np.array(restSensors)
        
        
    def extractData6Values(self, plateaus):
        returnList = []
        platList = []
        for i, arr in enumerate(plateaus):
            if arr != 0:
                platList.append(self.data6[i])
            elif len(platList) > 0:
                # List comp is used because it will copy by value a new array,
                # but just assigning platList was appending a reference to a list that's then cleared so returnList was empty.
                returnList.append([[int(plat[0]),int(plat[1]),int(plat[2]),int(plat[3])] for plat in platList])
                platList.clear()
        
        return returnList

    def avgSensorValues(self, plateaus):
        returnList = []

        for i, arr in enumerate(plateaus):
#            if np.shape(arr)[0] < 5 : # This value is an attempt to remove tiny plateaus
#                continue
            arr = np.array(arr)
            v1 = np.mean(arr[:,0]).astype(int)
            v2 = np.mean(arr[:,1]).astype(int)
            v3 = np.mean(arr[:,2]).astype(int)
            v4 = np.mean(arr[:,3]).astype(int)
            returnList.append([v1,v2,v3,v4])

        return returnList
   

    def plotAvgHighLows(self, avgPlateauValues, show = False):
                
        axisX = np.arange(len(avgPlateauValues))
        
        plt.title(self.fileName)
        plt.scatter(axisX, avgPlateauValues)
        if show:
            plt.show()
 
    def plotCopHighLows(self):
        plt.scatter([c.x for c in self.extensionPoints], [c.y for c in self.extensionPoints], color = 'r')
        plt.scatter([c.x for c in self.restPoints], [c.y for c in self.restPoints], color = 'g')
        plt.title(self.fileName + ' plot of the rest and extension points ')
        plt.show()
    
    def plotCopLine(self, show = True):
        
        plt.plot(self.copX, self.copY)
        plt.title(self.fileName)
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
    
        plt.title(self.fileName)
        
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
    
        plt.title(self.fileName)

        plt.scatter(axisX, tl, color = 'b')
        plt.scatter(axisX, tr, color = 'c')
        plt.scatter(axisX, bl, color = 'r')
        plt.scatter(axisX, br, color = 'y')
        
        if show:
            plt.show()
           
    # HINT Doesnt really work since I improved plateau generation
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
        with open('{}{}'.format(self.fileName, '.csv'), 'w') as file:
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
        