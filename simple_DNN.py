import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import Sequential
from keras.layers import Dense
from keras import callbacks

# Create some dummy data for training and testing
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 10))
y_test = np.random.randint(2, size=(100, 1))

# Model definition
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Add TensorBoard callback to visualize training progress
tensorboard_callback = callbacks.TensorBoard(log_dir='./logs')

# Model compilation
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training and TensorBoard callback
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(
    x_test, y_test), callbacks=[tensorboard_callback])

# Evaluation
score = model.evaluate(x_test, y_test, batch_size=16)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
