import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from AlexNet_layers import *

def get_Q1_model():
    
    model_input = keras.Input(shape=(227,227,3))
    model_output = keras.layers.Rescaling(1/255.)(model_input)
    model_output = keras.layers.Conv2D(96, (11,11),strides=4, activation='relu', 
                                        kernel_regularizer='l2',bias_regularizer='l2')(model_output)
    model_output = keras.layers.MaxPool2D( pool_size=(4, 4),strides=4)(model_output)
    model_output = keras.layers.Flatten()(model_output)
    model_output = keras.layers.BatchNormalization()(model_output)
    model_output = keras.layers.Dense(4096, activation='relu', )(model_output)
    model_output = keras.layers.BatchNormalization()(model_output)
    # model_output = keras.layers.Dropout(.3)(model_output)
    model_output = keras.layers.Dense(15, 
                                        kernel_regularizer='l2',bias_regularizer='l2')(model_output)
    model = keras.Model(model_input, model_output)
    
    return model



def get_Q2_model():
    model_input = keras.Input(shape=(227,227,3))
    model_output = keras.layers.Rescaling(1/255.)(model_input)
    
    out1 = keras.layers.Conv2D(48, (11,11),strides=4, activation='relu')(model_output)
    out2 = keras.layers.Conv2D(48, (11,11),strides=4, activation='relu')(model_output)

    out1 = keras.layers.MaxPool2D(pool_size=(3, 3),strides=2)(out1)
    out2 = keras.layers.MaxPool2D(pool_size=(3, 3),strides=2)(out2)

    out1 = keras.layers.Conv2D(128, (5,5),strides=1, activation='relu', padding='same')(out1)
    out2 = keras.layers.Conv2D(128, (5,5),strides=1, activation='relu', padding='same')(out2)

    out1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)(out1)
    out2 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)(out2)

    out1 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(out1)
    out2 = keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(out2)

    model_output = keras.layers.concatenate([out1, out2], axis = -1)
    model_output = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(model_output)
    
    model_output = keras.layers.Flatten()(model_output)
    model_output = keras.layers.Dropout(.3)(model_output)
    
    model_output = keras.layers.Dense(4096, activation='relu')(model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = keras.layers.Dense(4096, activation='relu')(model_output)
    model_output = keras.layers.Dropout(.5)(model_output)
    
    model_output = keras.layers.Dense(15)(model_output)
    
    model = keras.Model(model_input, model_output)
    return model


def get_Q3_model():
    model =  get_AlexNet_model()
    output = model.layers[-1].output
    output = keras.layers.Dense(15)(output)

    return keras.Model(model.layers[0].input, output)

   

def get_Q4_model():
    model = get_AlexNet_model()
    weights = get_AlexNet_weights()
    w_l_map = get_weight_layer_mapping()

    for key in w_l_map:
        for l in w_l_map[key]:
            model.layers[l].trainable = False
        # split weights of conv layers before setting
        if key[0] == 'c':
            w1, w2 = split_weights(weights=weights[key])
            model.layers[w_l_map[key][0]].set_weights(w1)
            model.layers[w_l_map[key][1]].set_weights(w2)
            continue
        model.layers[w_l_map[key][0]].set_weights(weights[key])

    output = model.layers[-1].output
    # output = keras.layers.BatchNormalization(momentum=.99)(output)
    output = keras.layers.LayerNormalization()(output)
    output = keras.layers.Dropout(.3)(output)
    output = keras.layers.Dense(15, kernel_regularizer=keras.regularizers.L2(1e-3),
                                    bias_regularizer=keras.regularizers.L2(1e-3))(output)


    return keras.Model(model.layers[0].input, output)

def get_Q5_model():
    model = get_Q4_model()

    model.load_weights('trained_models/2022-07-13_16_19_43.hdf5')

    w_l_map = get_weight_layer_mapping()
    for w in w_l_map:
        for l in w_l_map[w]:
            tmp = model.layers[l].weights[0].numpy()
            model.layers[l].trainable = True
            tmp -= model.layers[l].weights[0].numpy()
            if not (np.allclose(tmp, 0)):
                print('!')

    return model

