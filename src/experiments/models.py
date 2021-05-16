#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from src.ImageProcessing.CVasya import CVasya

NUM_CLASSES = 15
INPUT_SHAPE = (28, 28, 1)


def load_images():  # load test images from data
    return tfds.as_numpy(
        keras.preprocessing.image_dataset_from_directory(
            'src/experiments/data',
            labels='inferred',
            label_mode='int',
            color_mode="grayscale",
            batch_size=1000,
            image_size=(28, 28),
            seed=1337,
            interpolation='area'
        )
    )


def load_signs():  # load train signs from signs
    signs_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        )
    return signs_datagen.flow_from_directory(
        'src/experiments/signs',
        labels='inferred',
        label_mode='int',
        color_mode="grayscale",
        batch_size=2e5,
        target_size=(28, 28),
        seed=1337,
        interpolation='area'
    ).next()


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
    """SCORE: 0.875"""
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


def main():
    tensorflow.random.set_seed(1337)
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_signs, y_signs = load_signs()
    print(x_signs.shape)

    print(np.unique(y_train, return_counts=True))
    x_train, y_train = preprocess(np.concatenate((x_train, x_signs.reshape(-1, 28, 28)), axis=0),
                                  np.concatenate((y_train, y_signs + 10), axis=0))

    # model = load_model('src/experiments')
    model = Model4()
    model.fit(x_train, y_train)
    model.model.save('src/experiments')

    x_test, y_test = next(iter(load_images()))

    for i, x in enumerate(x_test):
        x_test[i] = CVasya.otsu(~x.astype(np.uint8)).reshape(INPUT_SHAPE)

    x_test, y_test = preprocess(x_test.reshape(-1, 28, 28), y_test)



if __name__ == '__main__':
    main()
