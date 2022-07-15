from train import *
from Logging import *

logger = Logger()

model, history = train_model(get_Q2_model, 
							# optimizer=tf.optimizers.SGD(learning_rate = .002, momentum=.5),
							epochs=20, 
							generators=lambda : get_generator(batch_size=64, augmentation=True))

save_trained_model(model, history, 2, logger)