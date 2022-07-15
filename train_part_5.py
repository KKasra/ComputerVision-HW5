from train import *
from Logging import *

logger = Logger()

model, history = train_model(get_Q5_model, 
							optimizer=tf.optimizers.SGD(learning_rate = .0001, momentum=.9),
							epochs=10, 
							generators=lambda : get_generator(batch_size=64, augmentation=True),
                            weights='trained_models/2022-07-15_12_10_44.hdf5'
                            )


save_trained_model(model, history, 4, logger)