#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from src.ImageProcessing.CVasya import CVasya

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def load_images(interpolation='area'):  # 'area' interpolation is the best one
    return tfds.as_numpy(
        keras.preprocessing.image_dataset_from_directory(
            'src/experiments/data',
            labels='inferred',
            label_mode='int',
            color_mode="grayscale",
            batch_size=1000,
            image_size=(28, 28),
            seed=1337,
            interpolation=interpolation
        )
    )


def preprocess(x, y):
    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    return x, y


class Model1:
    """SCORE: 0.8375"""
    """https://keras.io/examples/vision/mnist_convnet/"""

    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=INPUT_SHAPE),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

        self.batch_size = 128
        self.epochs = 15

        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)


class Model2:
    """REALLY BAD NN!!!"""
    """https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/"""

    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=INPUT_SHAPE),
                layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(NUM_CLASSES, activation='softmax')
            ]
        )

        self.batch_size = 128
        self.epochs = 10

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)


class Model3:
    """SCORE: 0.83745"""
    """https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/"""

    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=INPUT_SHAPE),
                layers.Conv2D(30, (5, 5), activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(15, (3, 3), activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(50, activation='relu'),
                layers.Dense(NUM_CLASSES, activation='softmax')
            ]
        )

        self.batch_size = 200
        self.epochs = 10

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)


class Model4:
    def __init__(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=INPUT_SHAPE),
                layers.Conv2D(7, (3, 3), activation='relu'),
                layers.Conv2D(14, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(14, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(28, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.MaxPooling2D((6, 6)),
                layers.Flatten(),
                layers.Dense(28, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(NUM_CLASSES, activation='softmax')
            ]
        )

        self.batch_size = 128
        self.epochs = 10

        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=1)


def main():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train, y_train = preprocess(x_train, y_train)

    x_test, y_test = next(iter(load_images()))
    for i, x in enumerate(x_test):
        x_test[i] = CVasya.otsu(~x.astype(np.uint8)).reshape(INPUT_SHAPE)
    x_test, y_test = preprocess(x_test.reshape(-1, 28, 28), y_test)

    models = [Model4()]

    for model in models:
        model.fit(x_train, y_train)
        print(model.score(x_test, y_test)[1])


if __name__ == '__main__':
    main()
