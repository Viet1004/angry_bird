# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:45:09 2021

@author: nguye
"""

import numpy as np
import matplotlib.pyplot as plt


theta = 1
DELTA_T = 0.001

sample = np.random.normal(theta,1, size = 1000)
sample_copy = sample.copy()
for j in range(1000):
    for i in np.arange(1,int(10/DELTA_T)):
        sample_copy[j] = (1-DELTA_T)*sample_copy[j] + DELTA_T*theta + np.sqrt(2*DELTA_T)*np.random.normal(0,1)
plt.figure()      
plt.hist(sample)
plt.hist(sample_copy)

