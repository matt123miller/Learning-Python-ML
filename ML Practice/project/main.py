# -*- coding: utf-8 -*-

import numpy as np
import requests
#import json

# Lets make some REST

apiRoot = "http://pokeapi.co/api/v2/"

response = requests.get(apiRoot + "pokemon")
rawOutput = response.json()
results = rawOutput["results"]

names = [p["name"] for p in results]
                
print (names)