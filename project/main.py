# -*- coding: utf-8 -*-

import numpy as np
import requests
import matplotlib.pyplot as plt

# Lets make some REST

apiRoot = "http://pokeapi.co/api/v2/"

response = requests.get(apiRoot + "pokemon")
rawOutput = response.json()
results = rawOutput["results"]

pokemon = [r for r in results]
names = [p["name"] for p in results]
chars = np.array([len(n) for n in names]) 

##print(pokemon)
print (names)
print(chars)
print(x)

# Make my graph axis
chars = np.sort(chars)          
x = np.array(range(0, len(chars)))     

#"""
plt.hist(chars)
plt.show()
#"""