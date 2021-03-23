#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from IPython.display import display
from keras.datasets import mnist
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def flatten(img):
    return img.copy().reshape(-1, 28 * 28)


class Model:

    def __init__(self):
        self.clf = None
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None

    def fit(self, X_train, Y_train, max_iter=100):
        self.X_train, self.Y_train = flatten(X_train).astype(np.float64), Y_train.astype(np.float64)
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter).fit(self.X_train, self.Y_train)

    def predict(self, X_test):
        self.X_test = flatten(X_test).astype(np.float64)
        return self.clf.predict(self.X_test)

    def score(self, X_test, Y_test):
        self.X_test, self.Y_test = flatten(X_test).astype(np.float64), Y_test.astype(np.float64)
        return self.clf.score(self.X_test, self.Y_test)


def resize(img, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, (28, 28), interpolation=interpolation)


def gray(img):
    return ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]


def otsu_resize_gray(img, interpolation=cv2.INTER_AREA):  # first gray, then resize
    return otsu(resize(gray(img), interpolation))


def otsu_gray_resize(img, interpolation=cv2.INTER_AREA):  # first resize, then gray
    return otsu(gray(resize(img, interpolation)))


def main():
    my_X_train, my_Y_train = [cv2.imread(f'./src/experiments/digits/{i}.png') for i in range(10)], np.arange(10)
    (mnist_X_train, mnist_Y_train), (mnist_X_test, mnist_Y_test) = mnist.load_data()
    stats_df = pd.DataFrame(columns=['mnist_transform', 'interpolation', 'my_transform', 'test_data', 'score'])
    for mnist_transform, mnist_transform_name in [(lambda img: img, 'no'), (otsu, 'otsu')]:
        model = Model()
        X_train, Y_train = np.array(list(map(mnist_transform, mnist_X_train))), mnist_Y_train
        model.fit(X_train, Y_train, max_iter=100)
        stats_df = stats_df.append({'mnist_transform': mnist_transform_name,
                                    'interpolation': 'no',
                                    'my_transform': 'no',
                                    'test_data': 'mnist',
                                    'score': model.score(np.array(list(map(mnist_transform, mnist_X_test))),
                                                         mnist_Y_test)},
                                   ignore_index=True)
        for interpolation, interpolation_name in zip(INTERPOLATIONS, ['nearest', 'linear', 'area', 'cubic', 'lanczos']):
            for my_transform, my_transform_name in [(lambda img: otsu_gray_resize(img, interpolation=interpolation), 'otsu_gray_resize'),
                                                    (lambda img: otsu_resize_gray(img, interpolation=interpolation), 'otsu_resize_gray')]:
                stats_df = stats_df.append({'mnist_transform': mnist_transform_name,
                                            'interpolation': interpolation_name,
                                            'my_transform': my_transform_name,
                                            'test_data': 'my',
                                            'score': model.score(np.array(list(map(my_transform, my_X_train))),
                                                                 my_Y_train)},
                                           ignore_index=True)
    display(stats_df)


if __name__ == '__main__':
    main()
