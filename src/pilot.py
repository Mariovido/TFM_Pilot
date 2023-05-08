import tensorflow as tf

from comet_ml import Experiment

import os
from dotenv import load_dotenv
from config import Config


# Initialization
load_dotenv()

PROJECT_COMET = os.getenv('PROJECT_COMET')
API_KEY_COMET = os.getenv('API_KEY_COMET')
WORKSPACE_COMET = os.getenv('WORKSPACE_COMET')

experiment = Experiment(
    api_key=API_KEY_COMET,
    project_name=PROJECT_COMET,
    workspace=WORKSPACE_COMET,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)

# Add parameters to the experiment
params = {
    "epochs": Config.epochs,
    "learning_rate": Config.learning_rate,
    "momentum": Config.momentum,
    "batch_size": Config.batch_size,
    "input_shape": Config.input_shape,
    "layer_1": Config.layer_1,
    "activation_1": Config.activation_1,
    "dropout": Config.dropout,
    "layer_2": Config.layer_2,
    "activation_2": Config.activation_2,
    "loss_function": Config.loss_function,
    "metric": Config.metric
}

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

# Callbacks creation
earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')

# Model training and log metrics with prefix 'train_'
with experiment.train():
    model.fit(x=x_train, y=y_train, epochs=Config.epochs,
              batch_size=Config.batch_size, validation_data=(x_test, y_test), callbacks=[earlyStopping])

# Log metrics with prefix 'test_'
with experiment.test():
    loss, accuracy = model.evaluate(x_test, y_test)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    experiment.log_metrics(metrics)

# End
experiment.log_parameters(params)
experiment.log_dataset_hash(x_train)
