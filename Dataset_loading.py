import tensorflow as tf
from tensorflow import keras
import numpy as np
img_height = 227
img_width = 227

normalize_batch = keras.layers.Normalization()

data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(.2),
    keras.layers.RandomZoom(1),
    keras.layers.RandomCrop(img_height,img_width)
  ]
    )

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

    if augmentation:
        train_ds.map(lambda x,y : (normalize_batch(data_augmentation(x)), y))
    else:
        train_ds.map(lambda x,y : (normalize_batch(x), y))
    test_ds.map(lambda x,y : (normalize_batch(x), y))

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds


