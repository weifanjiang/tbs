import numpy as np
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import pickle

cwd = os.getcwd()
if os.path.basename(cwd) == 'cnn-flowers':
    save_dir = cwd
else:
    save_dir = os.path.join(cwd, 'examples', 'cnn-flowers')

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# define the model
num_classes = len(class_names)

model = Sequential([
  layers.RandomFlip("horizontal",
                    input_shape=(img_height,
                                img_width,
                                3),
                    name='random_flip'),
  layers.RandomRotation(0.1, name='random_rotation'),
  layers.RandomZoom(0.1, name='random_zoom'),
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3), name='rescaling'),
  layers.Conv2D(16, 3, padding='same', activation='relu', name='conv1'),
  layers.MaxPooling2D(name='maxpool1'),
  layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2'),
  layers.MaxPooling2D(name='maxpool2'),
  layers.Dropout(0.2, name='drop_out'),
  layers.Flatten(name='flatten'),
  layers.Dense(128, activation='relu', name='dense'),
  layers.Dense(num_classes, name='output')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

with open(os.path.join(save_dir, 'model.pickle'), 'wb') as fout:
  pickle.dump(model, fout)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=None
)

test_datas = [x for x in test_ds]
X_data, Y_data = [x[0] for x in test_datas], [x[1] for x in test_datas]
X_data = np.array(X_data)

with open(os.path.join(save_dir, 'input.pickle'), 'wb') as fout:
  pickle.dump(X_data, fout)
