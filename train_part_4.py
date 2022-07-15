from train import *
from Logging import *

logger = Logger()

model, history = train_model(get_Q4_model, 
							# optimizer=tf.optimizers.SGD(learning_rate = .001, momentum=.9),
							epochs=10, 
							generators=lambda : get_generator(batch_size=128, augmentation=False)
                            )

save_trained_model(model, history, 4, logger)