import tensorflow as tf
from tensorflow import keras

def get_generator():
    img_height = 227
    img_width = 227
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/Train',
        validation_split=0,
        image_size=(img_height, img_width),
        batch_size=30)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/Test',
        validation_split=0,
        image_size=(img_height, img_width),
        batch_size=30)


    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds