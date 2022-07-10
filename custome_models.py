import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from AlexNet_layers import *

def get_Q1_model():
    layers = get_AlexNet_layers()
    
    model_input = keras.Input(shape=(227,227,3))
    model_output = keras.layers.Rescaling(1/255.)(model_input)
    model_output = layers['conv1'](model_output)
    model_output = keras.layers.MaxPool2D( pool_size=(4, 4),strides=4)(model_output)
    model_output = keras.layers.Flatten()(model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    model_output = layers['fc7'](model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    model_output = layers['fc8'](model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    model_output = keras.layers.Dense(15, activation='relu')(model_output)
#     model_output = keras.layers.Softmax()(model_output)
    model = keras.Model(model_input, model_output)
    
    del layers
    return model



def get_Q2_model():
    layers = get_AlexNet_layers()
    
    model_input = keras.Input(shape=(227,227,3))
    model_output = keras.layers.Rescaling(1/255.)(model_input)
    
    model_output = layers['conv1'](model_output)
    model_output = layers['maxp1'](model_output)
    
    model_output = layers['conv2'](model_output)
    model_output = layers['maxp2'](model_output)
    
    model_output = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(model_output)
    model_output = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(model_output)
    
    model_output = keras.layers.Flatten()(model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = layers['fc6'](model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = layers['fc7'](model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = layers['fc8'](model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = keras.layers.Dense(15, activation='relu')(model_output)
    
    model = keras.Model(model_input, model_output)
    del layers
    return model


