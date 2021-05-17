#!/usr/bin/env python3

import sympy
import cv2
import os

LABELS_TO_SYMBOLS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '/', 11: '==', 12: '-', 13: '+', 14: '*'
}


def solve(list_of_labels):
    try:
        return eval(''.join(map(lambda label: LABELS_TO_SYMBOLS[label], list_of_labels)))
    except SyntaxError:
        return False


if __name__ == '__main__':
    pass
