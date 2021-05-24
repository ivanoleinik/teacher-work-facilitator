import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key


class Image:
    """Library for image transforming."""

    @staticmethod
    def bgr_to_mnist(img, interpolation=cv2.INTER_AREA):
        gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = max(gray.shape[0:2])
        _, mnist_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
        f = np.zeros((s, s), np.uint8)
        ax, ay = (s - mnist_img.shape[1]) // 2, (s - mnist_img.shape[0]) // 2
        f[ay:mnist_img.shape[0] + ay, ax:ax + mnist_img.shape[1]] = mnist_img
        mnist_img = f
        res = cv2.resize(mnist_img, (28, 28), interpolation=interpolation)
        return res

    @staticmethod
    def mnist_filter(img):
        img_res = Image.filter_image_laplacian(img, is_bgr=True, blur_ker=(3, 3), c=1.3, block=3)
        kernel = (3, 3)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_DILATE, cv2.MORPH_ERODE]
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)
        return cv2.resize(img_res, dsize=(512, 512))


    @staticmethod
    def otsu(img, is_bgr=False, blur_ker=None):
        if is_bgr:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if blur_ker is not None:
            gray = cv2.blur(~gray, blur_ker)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)[1]

    @staticmethod
    def filter_image_using_hist(img, fileNum=None):
        plt.hist(img.ravel(), bins='rice')
        if fileNum is not None:
            plt.savefig(f'src/experiments/full_data/output/hist/{fileNum}.jpg')
            plt.close()

    @staticmethod
    def filter_image_laplacian(img, fileNum=None, is_bgr=True, c=1.3, blur_ker=(9, 9), block=11):
        block_agf = 11
        c_agf = 4.5

        if is_bgr:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if blur_ker is not None:
            gray = cv2.blur(gray, blur_ker)

        laplace = cv2.Laplacian(gray, cv2.CV_8UC1)
        bin_lap = cv2.adaptiveThreshold(laplace, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)

        return ~bin_lap

    @staticmethod
    def filter_image_adaptive_gaussian(img, is_bgr=True, c=4.5, blur_ker=(9, 9), block=11):
        block_agf = 11
        c_agf = 4.5

        if is_bgr:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if blur_ker is not None:
            gray = cv2.blur(gray, blur_ker)

        bin_agf = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_agf, c_agf)
        return ~bin_agf

    @staticmethod
    def filter_image_sobel(img, k=3):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        img = cv2.GaussianBlur(img, (9, 9), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=k, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=k, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        bin_grad = ~cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
        return bin_grad

    @staticmethod
    def _are_same_line(r1, r2):
        inter_length = max(min(r1[3], r2[3]) - max(r1[1], r2[1]), 0)
        return inter_length > min(r2[3] - r2[1], r1[3] - r1[1]) // 2

    @staticmethod
    def detect_digits(img, file_num=None):
        kernel = (5, 5)
        h, w = img.shape[:2]
        img_res = Image.filter_image_laplacian(img, file_num)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_CLOSE, cv2.MORPH_DILATE] * 3
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)

        contours, hierarchy = cv2.findContours(img_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = [cv2.boundingRect(c) for c in contours]

        def calculate_rect_points(r, w, h):
            x0, y0, dx, dy = r
            if not (w / 4 > dx > w / 100 and h / 4 > dy):
                return None

            x0 = max(0, int(x0 - dx * 0.04))
            x1 = min(w, int(x0 + dx * 1.08))
            y0 = max(0, int(y0 - dy * 0.04))
            y1 = min(h, int(y0 + dy * 1.08))

            if dx > dy:
                y0 = max(0, int(y0 - (dx - dy) * 0.54))
                if y0 == 0:
                    y1 = int(dx * 1.08)
                else:
                    y1 = min(h, int(y1 + (dx - dy) * 0.54))
                    if y1 == h:
                        y0 = int(h - dx * 1.08)

            return x0, y0, x1, y1

        def compare_rectangles(r1, r2):
            if Image._are_same_line(r1, r2):
                return r1[0] - r2[0]
            return r1[1] - r2[1]

        rect_points = [calculate_rect_points(r, w, h) for r in rects]
        return sorted(rect_points, key=cmp_to_key(compare_rectangles))

    @staticmethod
    def detect_lines(img):
        symbols = Image.detect_digits(img)
        line_start = 0
        lines = []
        for i, s in enumerate(symbols[:-1]):
            s1 = symbols[i + 1]
            if not Image._are_same_line(s, s1):
                lines.append(symbols[line_start:i + 1])
                line_start = i + 1
        lines.append(symbols[line_start:])
        return lines

    @staticmethod
    def cut_lines(img):
        lines = Image.detect_lines(img)
        img_lines = []
        for line in lines:
            img_lines.append([img[y0:y1, x0:x1, :] for x0, y0, x1, y1 in line])

        return img_lines

    @staticmethod
    def cut_digits(img):
        rects = Image.detect_digits(img)
        return [img[y0:y1, x0:x1, :] for x0, y0, x1, y1 in rects]
