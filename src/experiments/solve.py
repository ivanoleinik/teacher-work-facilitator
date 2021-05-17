#!/usr/bin/env python3

import sympy
import cv2
import os
import numpy as np
from src.ImageProcessing.CVasya import CVasya
from keras.models import load_model
from src.experiments.models import INPUT_SHAPE

LABELS_TO_SYMBOLS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '/', 11: '==', 12: '-', 13: '+', 14: '*'
}


def predict_image(img, model, proba=False):
    img = CVasya.bgr_to_mnist(img).reshape(INPUT_SHAPE)
    img = img.astype("float32") / 255
    img = np.array([img])
    return model.predict(img) if proba else np.argmax(model.predict(img))


def img_to_lines(img):
    lines = CVasya.cut_lines(img)
    model = load_model('src/experiments')
    res = []
    for line in lines:
        cur = []
        for img in line:
            cur.append(predict_image(img, model))
            print(predict_image(img, model, True))
        res.append(cur)
    return res


def join_list(list_of_labels):
    return ''.join(map(lambda label: LABELS_TO_SYMBOLS[label], list_of_labels))


def solve(list_of_labels):
    try:
        return eval(join_list(list_of_labels))
    except SyntaxError:
        return False


if __name__ == '__main__':
    img = cv2.imread('src/experiments/full_problem.jpg')
    lines = img_to_lines(img)
    for line in lines:
        print(join_list(line), solve(line))
