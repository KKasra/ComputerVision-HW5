import tensorflow as tf
from tensorflow import keras
import numpy as np
img_height = 227
img_width = 227

def normalize_batch(batch):
    return keras.layers.Normalization()(batch)

def data_augmentation(batch):
    model = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(.1),
    keras.layers.RandomZoom((-.2,0.2)),
    # keras.layers.RandomCrop(img_height,img_width)
  ]
    )
    x,y = np.random.uniform(low=0, high=1, size=(2,))

    x *= batch.shape[1] - img_height
    y *= batch.shape[2] - img_width

    x = int(x)
    y = int(y)
    return model(batch)
    # return model(batch)[:,x:x+img_height,y:y + img_width,:]

def get_generator(batch_size=30, augmentation=True):
    
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/AugmentedTrain' if augmentation else 'Data/Train',
        validation_split=0,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/Test',
        validation_split=0,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    train_ds.map(lambda x,y : (normalize_batch(data_augmentation(x)), y))
    test_ds.map(lambda x,y : (normalize_batch(x), y))

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds


