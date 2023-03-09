import tensorflow as tf


class Config:
    # Training hyperparameters
    epochs = 10
    learning_rate = 0.005
    momentum = 0.4
    test_size = 0.2
    batch_size = 64

    # DL algorithm
    optimizer = 'sgd'
    loss_function = tf.keras.losses.sparse_categorical_crossentropy.__name__
    metric = 'accuracy'

    # Model hyperparameters
    input_shape = (28, 28)
    layer_1 = 256
    activation_1 = tf.keras.activations.relu.__name__
    dropout = 0.5
    layer_2 = 10
    activation_2 = tf.keras.activations.softmax.__name__
