# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from point import Point

class Helper():

    @staticmethod
    def averagePoints(points):
        x, y = [], []
        for p in points:
            x.append(p.x)
            y.append(p.y)
        return Point(sum(x)/len(x),sum(y)/len(y))
    
    @staticmethod
    def splitDataAboveBelowMean(npIn, returnType):
        above = np.array([])
        below = np.array([])
        
        if returnType == 'm':
            mean = np.mean(npIn)
            above = npIn[npIn > mean]
            below = npIn[npIn < mean]
        else: #it's for points
            mean = Helper.averagePoints(npIn).sqrMagnitude()
#            for p in npIn:
#                sqrMag = p.sqrMagnitude()
#                if sqrMag > mean:
#                    above = np.append(above, p)
#                else:
#                    below = np.append(below, p)
            
            above = [p for p in npIn if p.sqrMagnitude() > mean]
            below = [p for p in npIn if p.sqrMagnitude() < mean]
        return above, below
    
    @staticmethod
    def pointListMinusPoint(points, point):
        rlist = []
        for p in points:
             rlist.append(p - point)
        return np.array(rlist)
    
    @staticmethod
    def normaliseOverHighestValue(values):
        outValues = []
        highest = np.max(values)
#        for v in values:
#            outValues.append(v / highest)
        return np.array([v / highest for v in values]) 
    
    
        