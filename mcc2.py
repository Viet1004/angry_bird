# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt



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

def perte_terminale(position):
    u1 = ((position[0] - D) - position[1]) / np.sqrt(2)
    u2 = ((position[0] - D) + position[1]) / np.sqrt(2)
    u3 = position[0] + position[1] - (D-15)
    return (u1 + u1 * (u1>0))**2 + u2**2 + (u3*(u3<0))**2

A = np.array([[0.9,0.2],
              [1,1.1]])

sumofAAT = A + A.T

inverseAAT = sp.linalg.inv(sumofAAT)

cov_vent = 25*inverseAAT*(np.array([[1,0],[0,1]]) - sp.linalg.expm(-sumofAAT*DELTA_T))
cov_vent



def dynamique_position_sur_un_pas_de_temps_round(position_actuelle, temps, vent_actuel_array, controle_actuel):
    position_actuelle = np.array(position_actuelle)
    controle_actuel = np.array(controle_actuel)
    terme1 = VITESSE_INIT * DELTA_T
    terme2 = G/2 * ((temps + DELTA_T)**2 - temps**2)
    terme3 = LAMBDA * position_actuelle * DELTA_T
    terme4 = vent_actuel_array * DELTA_T
    terme5 = controle_actuel * DELTA_T
    position_actuelle += terme1 + terme2 - terme3 + terme4 + terme5
    return position_actuelle

class control_model:
    def __init__(self,number_of_models,n_neurons):
      self.models = []
      self.distribution_info = []
      self.wind_collection = []
      for i in range(number_of_models):
        self.models.append(self._build_model(2,n_neurons))
      self._distribution_info()

    def _build_model(self, dim_data, n_neurons): 
      model = tf.keras.Sequential()
      model.add(layers.Dense(units = n_neurons, input_shape=(dim_data,), bias_initializer="glorot_uniform")) 
      model.add(layers.ELU())
      model.add(layers.Dense(units = n_neurons, input_shape=(dim_data,), bias_initializer="glorot_uniform"))
      model.add(layers.ELU())
      model.add(layers.Dense(units = 2, bias_initializer="glorot_uniform"))
      return model
    
    def wind_generation(number_sample):
      vent_collection = np.zeros((number_sample,int(2*T/DELTA_T)))
      for j in range(number_sample):        
        vents = np.zeros(int(2*T/DELTA_T))
        for i in range(int(T/DELTA_T) - 1):
            vents[2*(i+1):2*(i+2)] = sp.linalg.expm(-A*DELTA_T) @ vents[2*i:2*(i+1)] + np.random.multivariate_normal(mean = np.array([0,0]), cov = cov_vent)
        vent_collection[j] = vents.copy()
      return vent_collection

    def _wind_generator(self, time):
      return np.random.multivariate_normal(mean = np.zeros(2), cov = cov_vent)

    def next_step(self,position, control, time):
      for i in range(10):
        wind = self._wind_generator(time + i/10)
        position = dynamique_position_sur_un_pas_de_temps_round(position,time + i/10,wind,control)
      return position

    def model_access(self,i):
      return self.models[i]

    def train_step(self,i,position):
      with tf.GradientTape() as tape:
        control_predict = self.model_access(i)(position)
        loss_value = self.loss_function(i,position,control_predict)
      
      gradients = tape.gradient(loss_value, self.model_access(i).trainable_variables)
      return loss_value, gradients 

    def _loss_function_ind(self,time, position, control):
      loss = np.array(control).T @ np.array(control)
      temp = 0
      chosen_path = []
      if time == 9:
        for i in range(50):
          position_next = self.next_step(position, control, time)
          temp += perte_terminale(position_next)
        loss += temp/50
        return loss
      for next in np.arange(time+1, 10):
        temp = 0  
        if next == time +1:
          for i in range(50):
            position_next = self.next_step(position, control, next-1)
            control_next = self.model_access(next)(position_next)
            temp += control_next.T @ control_next
            if np.random.random() < 0.2:
              chosen_path.append((position_next,control_next))
          loss += temp/50

        else:
          temp_chosen_path = []
          for x in chosen_path:
              for i in range(5):
                position_next = self.next_step(x[0],x[1],next-1)
                control_next = self.model_access(next)(position_next)
                temp += control_next.T @ control_next 
                if np.random.random() < 0.2:
                  temp_chosen_path.append((position_next,control_next))
          loss += temp/(5*len(chosen_path))    
          chosen_path = temp_chosen_path
      temp = 0
      for x in chosen_path:
        for i in range(5):
          position_next = self.next_step(x[0],x[1], next-1)
          temp += perte_terminale(position_next)
      loss += temp/(5*len(chosen_path))
      print("Inside the function loss individuel: ",loss)
      return loss

    def loss_function(self, time, positions, controls):
      sum = 0
      size = len(positions)
      for i in range(size):
        sum += self._loss_function_ind(time, positions[i], controls[i])
      return sum/size

    def _distribution_info(self):
      with open("sample info.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
          mark = 0
          counter = 0
          value = []
          for i in range(len(line)):
            if line[i] == " " or line[i] == "\n":
              value.append(float("".join(line[mark:i])))
              mark = i+1
              counter +=1
              if counter == 5:
                break
          self.distribution_info.append((np.array([value[0],value[1]]),np.array([[value[2],value[3]],[value[3],value[4]]])))

    def _retrieve_distribution(self,time):
      return self.distribution_info[time]

    def train(self,i):
      list_loss = []
      EPOCHS = 1
      BATCH_SIZE = 4
      model = self.model_access(i)
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
      mean = self._retrieve_distribution(i-1)[0]
      cov = self._retrieve_distribution(i-1)[1]
      data = np.array([np.random.multivariate_normal(mean,cov) for j in range(16)])

      plt.scatter(data[:,0], data[:,1])
      
      
      for epoch in range(1,EPOCHS + 1):
        data_batches = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

        for data_batch in data_batches:
          loss_value, gradients = self.train_step(i, data_batch)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return model    
#        control_opt = np.array([self.model_access(i)(position) for position in validation_set])

model = control_model(10,6)

for i in range(10):
    model.train(9-i)
    
#for i in range(10):
#    model.model_access(i).save(f'model{i}.h5')

