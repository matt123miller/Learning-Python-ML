# -*- coding: utf-8 -*-

import numpy as np
import math
import random   

from participant import Participant



class KMeansClustering():
    def __init__(self, k=2, maxiterations=500):
        self.k = k
        self.maxIterations = maxiterations
        
    

def main():
    
    
    p = Participant(name = "0708 Trial1 TE", fileType = ".mat")
    print(p.filename)
    print(p.dataBlob)
    k = KMeansClustering(k=2)
    
    
    
    
    

if __name__ == "__main__":
    main()
    
    