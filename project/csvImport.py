# This is so simple!

import csv
import numpy as np

csvFile = open('Participant20.csv', newline='\n')

file = csv.reader(csvFile)

rows = {} 
i = 0

for row in file:
    rowdata = np.array(row[1:]).astype(float)
    rows[row[0]] = rowdata
    i += 1 # Couldn't work out how to range with the file       
             
print(rows.keys())
        
        
    