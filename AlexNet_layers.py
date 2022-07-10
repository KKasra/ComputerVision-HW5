import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_AlexNet_weights() :
    return np.load(open("tf_weights/bvlc_alexnet.npy", "rb"), 
                   encoding="latin1", allow_pickle=True).item()



def get_AlexNet_layers():
    conv1 = keras.layers.Conv2D(96, (11,11),strides=4, activation='relu')
    maxp1 = keras.layers.MaxPool2D( pool_size=(3, 3),strides=2)

    conv2 = keras.layers.Conv2D(256, (5,5),strides=1, activation='relu', padding='same')
    maxp2 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

    conv3 = keras.layers.Conv2D(384, (3,3), padding='same', activation='relu')

    conv4 = keras.layers.Conv2D(384, (3,3), padding='same', activation='relu')

    conv5 = keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')
    maxp5 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

    fc6 = keras.layers.Dense(4096, activation='relu')

    fc7 =  keras.layers.Dense(4096, activation='relu')
    
    fc8 = keras.layers.Dense(1000, activation='relu')
    
    result = dict()
    result['conv1'] = conv1
    result['maxp1'] = maxp1
    result['conv2'] = conv2
    result['maxp2'] = maxp2
    result['conv3'] = conv3
    result['conv4'] = conv4
    result['conv5'] = conv5
    result['maxp5'] = maxp5
    result['fc6'] = fc6
    result['fc7'] = fc7
    result['fc8'] = fc8
    return result