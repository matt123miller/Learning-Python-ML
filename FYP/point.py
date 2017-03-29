# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
class Point(object):
    
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)
        
    def sqrMagnitude(self):
        return (self.x ** 2 + self.y ** 2)
    
    def normalise(self):
        mag = self.magnitude()
        self.x = self.x / mag
        self.y = self.y / mag
        return self
        
    def __repr__(self):
        return "Point x:{0}, y:{1}".format(self.x,self.y)
     
    def printForUnity(self):
        return '{}, {}'.format(self.x.item(), self.y.item())
        
    def __add__(self,other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x,y)
        
    def __sub__(self,other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x,y)
        
    def __mul__(self,other):
        x = self.x * other.x
        y = self.y * other.y
        return Point(x,y)
    
    def __div__(self,other):
        x = self.x / other.x
        y = self.y / other.y
        return Point(x,y)
    
    def divideBy(self,scalar):
        return Point(self.x / scalar, self.y / scalar)
    
    @staticmethod
    def distance(x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
    
        return sqrt(distance)
    
    @staticmethod
    def averagePoints(points):
        x, y = [], []
        for p in points:
            x.append(p.x)
            y.append(p.y)
        return Point(sum(x)/len(x),sum(y)/len(y))
     
    @staticmethod
    def pointListMinusPoint(points, point):
        rlist = []
        for p in points:
             rlist.append(p - point)
        return np.array(rlist)
    
    @staticmethod
    def normaliseOverLongest(values):
        highest = np.max([v.magnitude() for v in values])
        return np.array([v.divideBy(highest) for v in values]) 