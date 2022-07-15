from train import *
from Logging import *

logger = Logger()

model, history = train_model(get_Q1_model, 
							optimizer=tf.optimizers.SGD(learning_rate = .002, momentum=.5),
							epochs=20, 
							generators=lambda : get_generator(batch_size=64, augmentation=False))

save_trained_model(model, history, 1, logger)