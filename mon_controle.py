import numpy as np
import scipy as sp

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from keras.models import load_model
import mcc2

def exemple_model(seconde, position):
    """
    un exemple naïf de contrôle avec la première composante le sinus du temps \
    et la deuxième le cosinus de la somme des coordonnées de poisiton
    """
    return np.array([np.sin(seconde), np.cos(position[0] + position[1])])


def main(seconde, position):
    """
    Votre controle du AG 2.0 au temps t seconde(s) et à la position X_t

    Parameters
    ----------
    seconde: float
        s = 0., 1., .., 9.
    position: arr (2,)
        Position du AG de format (2,) avec coordonnée x = position [0] et coordonnée y = position [1]

    Returns
    -------
    arr (2,)
        Contrôle du AG durant la prochaine seconde.

    Notes
    ----
    La sortie de cette fontion doit être la valeur de votre contrôle :
        un array (de float32 ou float64) et de dimension (2,)

    """
#    control = mcc2.model.model_access(seconde)(position)
    control = load_model("model{seconde}.h5")(position)   
    return control