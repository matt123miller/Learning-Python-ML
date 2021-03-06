#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:00:19 2017

@author: matthewmiller
"""

import numpy as np
from point import Point

## Recursive Feature Elimination
#from sklearn import datasets
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
## load the iris datasets
#dataset = datasets.load_iris()
## create a base classifier used to evaluate a subset of attributes
#model = LogisticRegression()
## create the RFE model and select 3 attributes
#rfe = RFE(model, 3)
#rfe = rfe.fit(dataset.data, dataset.target)
## summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)
##
#print(Point.normaliseOverLongest([Point(1,1), Point(2,2), Point(3,2), Point(10,5), Point(4,3), Point(6,6)]))

import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
print(np.shape(X))
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(probability=True, verbose=True)
clf.fit(X, y) 
print(clf.predict([[-0.8, -1]]))
