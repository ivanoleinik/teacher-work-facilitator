#!/usr/bin/env python3

import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


SEED = 1337
NUM_CLASSES = 15
INPUT_SHAPE = (28, 28, 1)


def load_signs(k=9):
    """Load train signs from signs directory."""
    signs_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    xs, ys = list(zip(*[signs_datagen.flow_from_directory(
        'data/signs',
        class_mode='categorical',
        color_mode="grayscale",
        batch_size=20000,
        target_size=(28, 28),
        seed=i,
    ).next() for i in range(k)]))
    return np.concatenate(xs), np.concatenate(ys)


def preprocess(x, y):
    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    return x, y


class Model:
    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=INPUT_SHAPE),
                layers.Conv2D(16, (3, 3), activation='relu'),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((3, 3)),
                layers.Dropout(0.25),

                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(NUM_CLASSES, activation='softmax')
            ]
        )

        self.batch_size = 128
        self.epochs = 15

        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       shuffle=True and not False,
                       epochs=self.epochs)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=1)


def build_model():
    tensorflow.random.set_seed(SEED)
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    print('Loading signs...')
    x_signs, y_signs = load_signs()
    y_signs = np.argmax(y_signs, axis=1) + 10

    x_train, y_train = preprocess(np.concatenate((x_train, x_signs.reshape(-1, 28, 28)), axis=0),
                                  np.concatenate((y_train, y_signs), axis=0))

    model = Model()
    print('Training...')
    model.fit(x_train, y_train)
    model.model.save('src/model')
