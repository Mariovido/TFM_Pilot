import tensorflow as tf
import numpy as np

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import os
from dotenv import load_dotenv
from config import Config


# Initialization
load_dotenv()
PROJECT_WANDB_PILOT = os.getenv('PROJECT_WANDB_PILOT')

wandb.init(project=PROJECT_WANDB_PILOT, config={
    'layer_1': Config.layer_1,
    'activation_1': Config.activation_1,
    'dropout': Config.dropout,
    'layer_2': Config.layer_2,
    'activation_2': Config.activation_2,
    'loss': Config.loss_function,
    'metric': Config.metric,
    "epoch": Config.epochs,
    'batch_size': Config.batch_size,
    'input_shape': Config.input_shape,
    'learning_rate': Config.learning_rate,
    'momentum': Config.momentum
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
    tf.keras.layers.Flatten(input_shape=Config.input_shape),
    tf.keras.layers.Dense(Config.layer_1, activation=Config.activation_1),
    tf.keras.layers.Dropout(Config.dropout),
    tf.keras.layers.Dense(Config.layer_2, activation=Config.activation_2)
])

# Create optimizer
optimizer = tf.keras.optimizers.SGD(
    learning_rate=Config.learning_rate, momentum=Config.momentum)

# Model compile
model.compile(optimizer=optimizer,
              loss=Config.loss_function, metrics=[Config.metric])

# Model training
model.fit(x=x_train, y=y_train, epochs=Config.epochs,
          batch_size=Config.batch_size, validation_data=(x_test, y_test), callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])

# End
wandb.finish()
