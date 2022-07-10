import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt

from custome_models import *
from Dataset_loading import get_generator
from Logging import Logger

def train_model(model_builder, epochs = 10, optimizer='adam', workers=1, multiprocessing_flag=False):
    model = model_builder()

    
    model.compile(optimizer=optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
             )
    model.summary()

    train_ds, test_ds = get_generator()

    history = model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=epochs,
      workers=workers,
     use_multiprocessing=multiprocessing_flag
    )
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    f, a = plt.subplots(1, 2)
    f.set_figwidth(12)
    f.set_figheight(8)
    
    a[0].plot(epochs_range, acc, label='Training Accuracy')
    a[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    
    a[1].plot(epochs_range, loss, label='Training Loss')
    a[1].plot(epochs_range, val_loss, label='Validation Loss')
    plt.savefig('tmp.jpg')

    return model, history



def save_trained_model(model, history, part, logger):

	print(part,
	 history.history['accuracy'][-1], history.history['val_accuracy'][-1],
	 history.history['loss'][-1], history.history['val_loss'][-1])


	timestamp = logger.log_model(part,
	 history.history['accuracy'][-1], history.history['val_accuracy'][-1],
	 history.history['loss'][-1], history.history['val_loss'][-1])

	file_path = 'train_models/{time}.h5'.format(time=timestamp)
	model.save(file_path)


logger = Logger()

model, history = train_model(get_Q2_model, optimizer=tf.optimizers.SGD(learning_rate=.05),
							epochs=1)

save_trained_model(model, history, 2, logger)