from train import *
from Logging import *

logger = Logger()

model, history = train_model(get_Q5_model, 
							# optimizer=tf.optimizers.SGD(learning_rate = .001, momentum=.9),
							epochs=10, 
							generators=lambda : get_generator(batch_size=128, augmentation=False),
                            weights="trained_models/2022-07-14_18_04_17.hdf5"
                            )


save_trained_model(model, history, 4, logger)