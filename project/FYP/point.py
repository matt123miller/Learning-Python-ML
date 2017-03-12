# -*- coding: utf-8 -*-
from math import sqrt

class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)
        
    def sqrMagnitude(self):
        return (self.x ** 2 + self.y ** 2)
        
    def __str__(self):
        return "Point(%s,%s)"%(self.x,self.y)
        
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