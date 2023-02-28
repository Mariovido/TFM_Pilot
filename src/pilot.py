import tensorflow as tf

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

import os
from dotenv import load_dotenv

# Initialization
load_dotenv()

PROJECT_NEPTUNE_PILOT = os.getenv('PROJECT_NEPTUNE_PILOT')
API_TOKEN_NEPTUNE_PILOT = os.getenv('API_TOKEN_NEPTUNE_PILOT')

run = neptune.init_run(
    project=PROJECT_NEPTUNE_PILOT,
    api_token=API_TOKEN_NEPTUNE_PILOT
)

# Parameters
learning_rate = 0.005
momentum = 0.4
epochs = 10
batch_size = 64

# Add parameters to the run
params = {"lr": learning_rate, "momentum": momentum,
          "epochs": epochs, "batch_size": batch_size}
run["parameters"] = params

# Dataset initialization
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

optimizer = tf.keras.optimizers.SGD(
    learning_rate=params['lr'], momentum=params["momentum"])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Neptune callback initialization
neptune_cbk = NeptuneCallback(run=run, base_namespace='training')

# Model training
model.fit(x_train, y_train, epochs=params['epochs'],
          batch_size=params['batch_size'], callbacks=neptune_cbk)

# Model evaluation
eval_metrics = model.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    run['eval/{}'.format(model.metrics_names[j])] = metric

# End
run.stop()
