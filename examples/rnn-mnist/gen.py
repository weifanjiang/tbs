# Importing libraries
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import os
import pickle

# Setting variables.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device_str = '/CPU:0'

cwd = os.getcwd()
if os.path.basename(cwd) == 'rnn-mnist':
    save_dir = cwd
else:
    save_dir = os.path.join(cwd, 'examples', 'rnn-mnist')

# Loading the MNIST dataset. Caching is handled by Keras.
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
NUM_LABELS = len(np.unique(y_train))

# Data Processing Step
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
image_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, image_size, image_size]).astype('float32') / 255
X_test = np.reshape(X_test, [-1, image_size, image_size]).astype('float32') / 255

# Setting variables for ML training pipeline
INPUT_SIZE = tuple([image_size, image_size])
BATCH_SIZE = 128
EPOCHS = 5

# Defining the ML model
model = keras.Sequential([
    layers.SimpleRNN(128, input_shape=INPUT_SIZE, dropout=0.2, name='rnn'),
    layers.Dense(NUM_LABELS, activation='softmax', name='output')
])

# Compling the model.
model.compile(optimizer="sgd", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"]
             )  

history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS
)

# Evaluating the model.
_, acc = model.evaluate(X_test,
                        y_test,
                        batch_size=BATCH_SIZE,
                        verbose=0)

print(f"Test accuracy: {round(acc * 100, 2)} %.")

# Saving the model.
with open(os.path.join(save_dir, 'model.pickle'), 'wb') as fout:
    pickle.dump(model, fout)

with open(os.path.join(save_dir, 'input.pickle'), 'wb') as fout:
    pickle.dump(X_test, fout)
