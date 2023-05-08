import tensorflow as tf

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

import os
from dotenv import load_dotenv
from config import Config

# Initialization
load_dotenv()

PROJECT_NEPTUNE = os.getenv('PROJECT_NEPTUNE')
API_TOKEN_NEPTUNE = os.getenv('API_TOKEN_NEPTUNE')

run = neptune.init_run(
    project=PROJECT_NEPTUNE,
    api_token=API_TOKEN_NEPTUNE
)

# Add parameters to the run
params = {"lr": Config.learning_rate, "momentum": Config.momentum, "epochs": Config.epochs,
          "batch_size": Config.batch_size, "layer_1": Config.layer_1, "activation_1": Config.activation_1,
          "dropout": Config.dropout, "layer_2": Config.layer_2, "activation_2": Config.activation_2,
          "loss_function": Config.loss_function, "metric": Config.metric, "input_shape": str(Config.input_shape)}
run["parameters"] = params

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

optimizer = tf.keras.optimizers.SGD(
    learning_rate=Config.learning_rate, momentum=Config.momentum)

model.compile(optimizer=optimizer, loss=Config.loss_function,
              metrics=Config.metric)

# Neptune callback initialization
neptune_cbk = NeptuneCallback(run=run, base_namespace='training')

# Model training
model.fit(x_train, y_train, epochs=Config.epochs,
          batch_size=Config.batch_size, callbacks=neptune_cbk)

# Model evaluation
eval_metrics = model.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    run['eval/{}'.format(model.metrics_names[j])] = metric

# End
run.stop()
