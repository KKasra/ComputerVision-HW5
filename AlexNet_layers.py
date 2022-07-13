import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_AlexNet_weights() :
    return np.load(open("tf_weights/bvlc_alexnet.npy", "rb"), 
                   encoding="latin1", allow_pickle=True).item()

def split_weights(weights):
    n = weights[1].shape[0] // 2
    return [[weights[0][:,:,:,:n], weights[1][:n]], [weights[0][:,:,:,n:], weights[1][n:]]]
def get_weight_layer_mapping():
    weight_layer_map = dict()

    weight_layer_map['conv1'] = [2, 3]
    weight_layer_map['conv2'] = [6, 7]
    weight_layer_map['conv3'] = [10, 11]
    weight_layer_map['conv4'] = [12, 13]
    weight_layer_map['conv5'] = [14, 15]
    weight_layer_map['fc6'] = [19]
    weight_layer_map['fc7'] = [20]

    return weight_layer_map

def get_AlexNet_model():
    model_input = keras.Input(shape=(227,227,3))
    model_output = keras.layers.Rescaling(1/1.)(model_input)
    
    out1 = keras.layers.Conv2D(48, (11,11),strides=4, activation='relu')(model_output)
    out2 = keras.layers.Conv2D(48, (11,11),strides=4, activation='relu')(model_output)
    
    out1 = keras.layers.MaxPool2D(pool_size=(3, 3),strides=2)(out1)
    out2 = keras.layers.MaxPool2D(pool_size=(3, 3),strides=2)(out2)
    

    out1 = keras.layers.Conv2D(128, (5,5),strides=1, activation='relu', padding='same')(out1)
    out2 = keras.layers.Conv2D(128, (5,5),strides=1, activation='relu', padding='same')(out2)

    model_output = keras.layers.concatenate([out1, out2], axis=-1)
    model_output = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)(model_output)

    out1 = keras.layers.Conv2D(192, (3,3), padding='same', activation='relu')(model_output)
    out2 = keras.layers.Conv2D(192, (3,3), padding='same', activation='relu')(model_output)

    out1 = keras.layers.Conv2D(192, (3,3), padding='same', activation='relu')(out1)
    out2 = keras.layers.Conv2D(192, (3,3), padding='same', activation='relu')(out2)

    out1 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(out1)
    out2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(out2)


    model_output = keras.layers.concatenate([out1, out2], axis = -1)

    model_output = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)(model_output)

    model_output = keras.layers.Flatten()(model_output)
 
    model_output = keras.layers.Dense(4096, activation='relu')(model_output)

    model_output = keras.layers.Dense(4096, activation='relu')(model_output)

    model = keras.Model(model_input, model_output)

    return model