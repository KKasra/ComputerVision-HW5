import tensorflow as tf
from tensorflow import keras

def get_generator(batch_size=30, augmentation=True):
    img_height = 227
    img_width = 227
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/AugmentedTrain' if augmentation else 'Data/Train',
        validation_split=0,
        image_size=(img_height, img_width),
        # crop_to_aspect_ratio=True,
        batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/Test',
        validation_split=0,
        image_size=(img_height, img_width),
        # crop_to_aspect_ratio=True,
        batch_size=batch_size)


    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds