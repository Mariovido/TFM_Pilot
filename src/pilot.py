import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf

import os
from dotenv import load_dotenv

# Initialization
load_dotenv()
PROJECT_WANDB_PILOT = os.getenv('PROJECT_WANDB_PILOT')

wandb.init(project=PROJECT_WANDB_PILOT, config={
    'layer_1': 512,
    'activation_1': 'relu',
    'dropout': random.uniform(0.01, 0.80),
    'layer_2': 10,
    'activation_2': 'softmax',
    'optimizer': 'sgd',
    'loss': 'sparse_categorical_crossentropy',
    'metric': 'accuracy',
    "epoch": 8,
    'batch_size': 256
})

# Use wandb.config as config
config = wandb.config

# Dataset initialization
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train = x_train[::5], y_train[::5]
x_test, y_test = x_test[::20], y_test[::20]
labels = [str(digit) for digit in range(np.max(y_train) + 1)]

# Model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
    tf.keras.layers.Dropout(config.dropout),
    tf.keras.layers.Dense(config.layer_2, activation=config.activation_2)
])

# Model compile
model.compile(optimizer=config.optimizer,
              loss=config.loss, metrics=[config.metric])

# Model training
model.fit(x=x_train, y=y_train, epochs=config.epoch,
          batch_size=config.batch_size, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])

# End
wandb.finish()
