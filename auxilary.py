# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:04:05 2021

@author: nguye
"""


import numpy as np
import pandas as pd
import scipy as sp
import argparse
import sys
import controle_etudiant as ce
import matplotlib.pyplot as plt
import csv
D = 200  # distance en metre au temps T=10 secondes
T = 10  # temps en seconde
DELTA_T = 0.1  # pas de temps pour discrétiser en seconde
G = np.array([0., -4])  # force de gravité en m/s^2
LAMBDA = 0.01  # coefficient de resitance de l'air en kg/s

POSITION_INIT_x = 0.  # coordonnée x de la position initiale
POSITION_INIT_y = 0.  # coordonnée y de la position initiale

VITESSE_INIT_x = D / T  # coordonnée x en m/s de la vitesse initiale
VITESSE_INIT_y = (T/2) * np.linalg.norm(G)  # coordonnée y en m/s de la vitesse initiale
VITESSE_INIT = np.array([VITESSE_INIT_x, VITESSE_INIT_y])  # vitesse initiale sous forme d'array

position_actuelle = np.array([POSITION_INIT_x, POSITION_INIT_y])  # position initiale
cout_controle_trajectoriel = 0.  # initialisation du cout de controle



A = np.array([[0.9,0.2],
              [1,1.1]])

sumofAAT = A + A.T

inverseAAT = sp.linalg.inv(sumofAAT)

cov_vent = 25*inverseAAT*(np.array([[1,0],[0,1]]) - sp.linalg.expm(-sumofAAT*DELTA_T))

print(cov_vent)

def dynamique_position_sur_un_pas_de_temps(position_actuelle, temps, vent_actuel_array, controle_actuel):
    """
    Dynamique de la position discrétisée au pas DELTA_T seconde entre deux pas de temps t_i et t_i+1

    Parameters
    ----------
    position_actuelle: arr (2,)
    temps: float
    vent_actuel_array: arr (2, )
    controle_actuel: arr (2,)

    Returns
    -------
    arr (2,)
        Position au temps t_i+1 = t_i + DELTA_T
    """

    terme1 = VITESSE_INIT * DELTA_T
    terme2 = G/2 * ((temps + DELTA_T)**2 - temps**2)
    terme4 = vent_actuel_array * DELTA_T
    terme5 = controle_actuel * DELTA_T
    return position_actuelle + terme1 + terme2 + terme4 + terme5

SAMPLE_SIZE = 100
def wind_generation(number_sample):
    vent_collection = np.zeros((number_sample,int(2*T/DELTA_T)))
    for j in range(number_sample):        
        vents = np.zeros(int(2*T/DELTA_T))
        for i in range(int(T/DELTA_T) - 1):
            vents[2*(i+1):2*(i+2)] = sp.linalg.expm(-A*DELTA_T) @ vents[2*i:2*(i+1)] + np.random.multivariate_normal(mean = np.array([0,0]), cov = cov_vent)
        vent_collection[j] = vents.copy()
    return vent_collection

wind_collection = wind_generation(SAMPLE_SIZE)
def distribution(time, position):
    positions = np.array([position for i in range(SAMPLE_SIZE)])
    for j in range(10):
        for i in range(SAMPLE_SIZE):    
            positions[i] = dynamique_position_sur_un_pas_de_temps(positions[i],time + j/10, wind_collection[ i][time*20 + j*2:time*20 + (j+1)*2],0)
    return positions



with open("sample info.txt","w+") as f:  
    for i in range(10):
        sample = distribution(i, np.array([20*i, 20*i-2*i**2], dtype = 'float'))
        print(np.shape(sample))
        mean = sample.mean(axis = 0)
        print(mean)
        cov = np.cov(sample.T)
        print("cov is: ",cov)

        f.write(f"{mean[0],mean[1],cov[0][0],cov[0][1],cov[1][1]} \n")
        
        plt.figure()
        plt.scatter(sample[:,0], sample[:,1])   
    
    
    