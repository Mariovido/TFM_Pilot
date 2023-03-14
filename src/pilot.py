import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import os
from dotenv import load_dotenv
from config import Config

import datetime

# Initialization
load_dotenv()

# Parameters initialization
EPOCHS = hp.HParam('epochs', hp.Discrete([Config.epochs]))
LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([Config.learning_rate]))
MOMENTUM = hp.HParam('momentum', hp.Discrete([Config.momentum]))
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([Config.batch_size]))
LOSS_FUNCTION = hp.HParam('loss_function', hp.Discrete([Config.loss_function]))
METRIC = hp.Metric(Config.metric, display_name=Config.metric)
INPUT_SHAPE = hp.HParam('input_shape', hp.Discrete([str(Config.input_shape)]))
LAYER_1 = hp.HParam('layer_1', hp.Discrete([Config.layer_1]))
ACTIVATION_1 = hp.HParam('activation_1', hp.Discrete([Config.activation_1]))
DROPOUT = hp.HParam('dropout', hp.Discrete([Config.dropout]))
LAYER_2 = hp.HParam('layer_2', hp.Discrete([Config.layer_2]))
ACTIVATION_2 = hp.HParam('activation_2', hp.Discrete([Config.activation_2]))

# Initialize file creation
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir_hparams = 'logs/hparam_tuning/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

with tf.summary.create_file_writer(log_dir_hparams).as_default():
    # Hyperparameters setting
    HPARAMS = [EPOCHS, LEARNING_RATE, MOMENTUM, BATCH_SIZE, LOSS_FUNCTION,
               INPUT_SHAPE, LAYER_1, ACTIVATION_1, DROPOUT, LAYER_2, ACTIVATION_2]
    hp.hparams_config(hparams=HPARAMS, metrics=[METRIC])

    hparams = {
        EPOCHS: Config.epochs,
        LEARNING_RATE: Config.learning_rate,
        MOMENTUM: Config.momentum,
        BATCH_SIZE: Config.batch_size,
        LOSS_FUNCTION: Config.loss_function,
        INPUT_SHAPE: str(Config.input_shape),
        LAYER_1: Config.layer_1,
        ACTIVATION_1: Config.activation_1,
        DROPOUT: Config.dropout,
        LAYER_2: Config.layer_2,
        ACTIVATION_2: Config.activation_2
    }
    hp.hparams(hparams)

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

    # TensorBoard callback initialization
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    keras_callback = hp.KerasCallback(log_dir, hparams=hparams)

    # Model training
    model.fit(x=x_train, y=y_train, epochs=Config.epochs,
              batch_size=Config.batch_size, validation_data=(x_test, y_test), callbacks=[tensorboard_callback, keras_callback])

    # Model evaluation
    _, accuracy = model.evaluate(x_test, y_test)

    # End
    tf.summary.scalar(Config.metric, accuracy, step=1)
