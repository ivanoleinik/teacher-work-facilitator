#!/usr/bin/env python3

import numpy as np
import pandas as pd
from IPython.display import display
from keras.datasets import mnist
from src.ImageProcessing.CVasya import CVasya
from src.experiments.models import Model1, Model3, Model4, load_images, preprocess, INPUT_SHAPE


def main():
    stats_df = pd.DataFrame(columns=['model', 'score'])

    (x_train, y_train), _ = mnist.load_data()
    x_train, y_train = preprocess(x_train, y_train)

    x_test, y_test = next(iter(load_images()))
    for i, x in enumerate(x_test):
        x_test[i] = CVasya.otsu(~x.astype(np.uint8)).reshape(INPUT_SHAPE)
    x_test, y_test = preprocess(x_test.reshape(-1, 28, 28), y_test)

    models = [Model1(), Model4()]
    for model in models:
        model.fit(x_train, y_train)
        stats_df = stats_df.append({'model': model.__class__.__name__,
                                    'score': model.score(x_test, y_test)[1]},
                                   ignore_index=True)
    display(stats_df)


if __name__ == '__main__':
    main()
