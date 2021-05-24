#!/usr/bin/env python3

import sympy
import cv2
import os
import numpy as np
from src.ImageProcessing.CVasya import CVasya
from keras.models import load_model
from src.experiments.models import INPUT_SHAPE
from itertools import product

PROBA_THRESHOLD = 0.01

LABELS_TO_SYMBOLS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '/', 11: '==', 12: '-', 13: '+', 14: '*'
}


def predict_image(img, model, proba=False):
    img = CVasya.bgr_to_mnist(img).reshape(INPUT_SHAPE)
    # cv2.imshow('', img)
    # cv2.waitKey(0)
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
            cur.append(predict_image(img, model, proba=True))
        res.append(cur)
    return res


def join_list(list_of_labels):
    return ''.join(map(lambda label: LABELS_TO_SYMBOLS[label], list_of_labels))


def solve(list_of_labels):
    try:
        return eval(join_list(list_of_labels))
    except SyntaxError:
        return None
    except ZeroDivisionError:
        return None


def main():
    img = cv2.imread('src/experiments/full_problem.jpg')
    lines = img_to_lines(img)
    for line in lines:
        variants = []
        for symbol_proba in line:
            symbol_proba = symbol_proba.ravel()
            competitive = {}
            for label in LABELS_TO_SYMBOLS:
                if symbol_proba[label] > PROBA_THRESHOLD:
                    competitive[label] = symbol_proba[label]
            variants.append(competitive)
        # print(variants)
        probas = []
        for keys in product(*[list(d.keys()) for d in variants]):
            proba = np.prod([variants[i][key] for i, key in enumerate(keys)])
            res = solve(keys)
            if res is not None:
                probas.append((proba, res, join_list(keys)))
        probas.sort()
        probas = probas[::-1]
        true, false = 0.0, 0.0
        print(probas[:3])
        for proba, res, _ in probas:
            if res:
                true += proba
                break
            else:
                false += proba
        print(f'Truth proba: {true / (true + false)}')


if __name__ == '__main__':
   main()
