import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt

from custome_models import *
from Dataset_loading import get_generator
from Logging import Logger

def train_model(model_builder, epochs = 10, optimizer='adam', 
					workers=1, multiprocessing_flag=False,generators=get_generator, weights = None):
    model = model_builder()

    
    model.compile(optimizer=optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
             )

    if weights is not None:
        model.load_weights(weights)

    

    train_ds, test_ds = generators()

    

    if weights is not None:
        history = model.fit(
		train_ds,
		validation_data=test_ds,
		epochs=1,
		workers=workers,
		use_multiprocessing=multiprocessing_flag
		)
        model.load_weights(weights)
        for l in model.layers:
            l.trainable = True
        print(len(model.trainable_variables))
        

    model.summary()

    history = model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=epochs,
      workers=workers,
     use_multiprocessing=multiprocessing_flag
    )
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    acc5 = history.history['sparse_top_k_categorical_accuracy']
    val_acc5 = history.history['val_sparse_top_k_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    f, a = plt.subplots(3, 1)
    f.set_figwidth(16)
    f.set_figheight(16)
    
    a[0].plot(epochs_range, acc, label='Training Accuracy')
    a[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    a[0].legend()
    a[0].grid()
    
    a[1].plot(epochs_range, acc5, label='Training Top-5 Accuracy')
    a[1].plot(epochs_range, val_acc5, label='Validation Top-5 Accuracy')
    a[1].legend()
    a[1].grid()
    a[2].plot(epochs_range, loss, label='Training Loss')
    a[2].plot(epochs_range, val_loss, label='Validation Loss')
    a[2].legend()
    a[2].grid()
    plt.savefig('tmp.jpg')

    return model, [acc[-1], val_acc[-1], acc5[-1], val_acc5[-1], loss[-1], val_loss[-1]]



def save_trained_model(model, history, part, logger):
	timestamp = logger.log_model(part,history)

	file_path = 'trained_models/{time}.hdf5'.format(time=timestamp.replace(' ', '_').replace(':', '_'))
	model.save_weights(file_path)


# logger = Logger()

# get_Q4_model().summary()

# model, history = train_model(get_Q4_model, 
# 							optimizer=tf.optimizers.SGD(learning_rate = .002, momentum=.5),
# 							epochs=20, 
# 							generators=get_generator(batch_size=64, augmentation=True))

# save_trained_model(model, history, 1, logger)