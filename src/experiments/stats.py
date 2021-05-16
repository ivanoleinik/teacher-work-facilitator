#!/usr/bin/env python3

import numpy as np
import pandas as pd
from IPython.display import display
from keras.datasets import mnist
from keras.models import load_model
from src.ImageProcessing.CVasya import CVasya
from src.experiments.models import load_images, load_signs, preprocess, INPUT_SHAPE


def main():
    stats_df = pd.DataFrame(columns=['score'])

    # (x_train, y_train), _ = mnist.load_data()
    # x_train, y_train = preprocess(x_train, y_train)

    x_test, y_test = next(iter(load_images()))

    for i, x in enumerate(x_test):
        x_test[i] = CVasya.otsu(~x.astype(np.uint8)).reshape(INPUT_SHAPE)
    x_test, y_test = preprocess(x_test.reshape(-1, 28, 28), y_test)
    print(x_test.shape)
    model = load_model('src/experiments')
    print(model.predict(x_test))
    stats_df = stats_df.append({'score': model.evaluate(x_test, y_test)[1]},
                               ignore_index=True)
    display(stats_df)


if __name__ == '__main__':
    main()
