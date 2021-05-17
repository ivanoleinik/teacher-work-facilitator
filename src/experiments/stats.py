#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
from src.experiments import solve
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

    model = load_model('src/experiments')
    prediction = np.argmax(model.predict(x_test), axis=1)
    for i, (pred, true) in enumerate(zip(prediction, y_test)):
        true = np.argmax(true)
        if pred != true:
            cv2.imwrite(f'src/experiments/wrong/i_{solve.LABELS_TO_SYMBOLS[pred]}_{true}.png', 255 * x_test[i])
    # stats_df = stats_df.append({'score': model.evaluate(x_test, y_test)[1]},
    #                            ignore_index=True)

    # display(stats_df)


if __name__ == '__main__':
    main()
