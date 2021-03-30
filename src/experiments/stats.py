#!/usr/bin/env python3

import numpy as np
import pandas as pd
from IPython.display import display
from keras.datasets import mnist
from src.ImageProcessing.Cvasya import Cvasya
from src.experiments.models import Model1, Model3, load_images, preprocess, INPUT_SHAPE

INTERPOLATIONS = ['bilinear', 'nearest', 'bicubic', 'area',
                  'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic']


def main():
    stats_df = pd.DataFrame(columns=['interpolation', 'model', 'score'])

    (x_train, y_train), _ = mnist.load_data()
    x_train, y_train = preprocess(x_train, y_train)

    model1, model3 = Model1(), Model3()
    model1.fit(x_train, y_train)
    model3.fit(x_train, y_train)

    for interpolation in INTERPOLATIONS:
        x_test, y_test = next(iter(load_images(interpolation=interpolation)))
        for i, x in enumerate(x_test):
            x_test[i] = Cvasya.otsu(~x.astype(np.uint8)).reshape(INPUT_SHAPE)
        x_test, y_test = preprocess(x_test.reshape(-1, 28, 28), y_test)
        stats_df = stats_df.append({'interpolation': interpolation,
                                    'model': 'Model1',
                                    'score': model1.score(x_test, y_test)[1]},
                                   ignore_index=True)
        stats_df = stats_df.append({'interpolation': interpolation,
                                    'model': 'Model3',
                                    'score': model3.score(x_test, y_test)[1]},
                                   ignore_index=True)
    display(stats_df)


if __name__ == '__main__':
    main()
