import tensorflow as tf

import os
from dotenv import load_dotenv
from config import Config


# Initialization
load_dotenv()

# Hyperparameters
epochs = Config.epochs
learning_rate = Config.learning_rate
momentum = Config.momentum
batch_size = Config.batch_size
loss_function = Config.loss_function
metric = Config.metric
input_shape = Config.input_shape
layer_1 = Config.layer_1
activation_1 = Config.activation_1
dropout = Config.dropout
layer_2 = Config.layer_2
activation_2 = Config.activation_2

# Dataset initialization
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=Config.input_shape),
    tf.keras.layers.Dense(Config.layer_1, activation=Config.activation_1),
    tf.keras.layers.Dropout(Config.dropout),
    tf.keras.layers.Dense(Config.layer_2, activation=Config.activation_2)
])

# Optimizer initialization
optimizer = tf.keras.optimizers.SGD(
    learning_rate=Config.learning_rate, momentum=Config.momentum)

# Model compile
model.compile(loss=Config.loss_function,
              optimizer=optimizer, metrics=[Config.metric])

# Model training
model.fit(x=x_train, y=y_train, epochs=Config.epochs,
          batch_size=Config.batch_size, validation_data=(x_test, y_test))
