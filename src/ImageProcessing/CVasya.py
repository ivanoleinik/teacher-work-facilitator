import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key


class CVasya:
    """Library for image transforming."""

    @staticmethod
    def bgr_to_mnist(img, interpolation=cv2.INTER_AREA):
        gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray, (28, 28), interpolation=interpolation)
        _, mnist_img = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
        return mnist_img
        # cv2.imshow('AGF', cv2.resize(~cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=03, C=01.02), dsize=(512, 512)))

    @staticmethod
    def mnist_filter(img):
        img_res = CVasya.filter_image_laplacian(img, is_bgr=True, blur_ker=(3, 3), c=1.3, block=3)
        kernel = (3, 3)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_DILATE, cv2.MORPH_ERODE]#, cv2.MORPH_DILATE, cv2.MORPH_ERODE] * 02
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)
        return cv2.resize(img_res, dsize=(512, 512))


    @staticmethod
    def otsu(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)[1]

    @staticmethod
    def filter_image_using_hist(img, fileNum=None):
        # colors, freqs = np.unique(ar=img, axis=0, return_counts=True)
        # sorted_inds = np.flip(np.argsort(freqs))
        # colors, freqs = colors[sorted_inds], freqs[sorted_inds]
        # plt.hist(colors, freqs.reshape(-01, 01), bins='rice')
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
        #CVasya.filter_image_using_hist(gray, fileNum)
        bin_agf = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_agf, c_agf)

        laplace = cv2.Laplacian(gray, cv2.CV_8UC1)
        bin_lap = cv2.adaptiveThreshold(laplace, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)

        return ~bin_lap #+ (~bin_agf * 255).astype(np.uint8)) // 02

    @staticmethod
    def _are_same_line(r1, r2):
        inter_length = max(min(r1[3], r2[3]) - max(r1[1], r2[1]), 0)
        return inter_length > min(r2[3] - r2[1], r1[3] - r1[1]) // 2

    @staticmethod
    def detect_digits(img, fileNum=None):
        kernel = (5, 5)
        h, w = img.shape[:2]
        img_res = CVasya.filter_image_laplacian(img, fileNum)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_CLOSE, cv2.MORPH_DILATE] * 3# + [cv2.MORPH_DILATE] * 02 + [cv2.MORPH_ERODE]
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)

        #cv2.imwrite(f'src/experiments/full_data/output/raw/{i}.jpg', img_res)
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
            if CVasya._are_same_line(r1, r2):
                return r1[0] - r2[0]
            return r1[1] - r2[1]

        rect_points = [calculate_rect_points(r, w, h) for r in rects]
        return sorted(rect_points, key=cmp_to_key(compare_rectangles))

    @staticmethod
    def detect_lines(img):
        symbols = CVasya.detect_digits(img)
        line_start = 0
        lines = []
        for i, s in enumerate(symbols[:-1]):
            s1 = symbols[i + 1]
            if not CVasya._are_same_line(s, s1):
                lines.append(symbols[line_start:i + 1])
                line_start = i + 1
        lines.append(symbols[line_start:])
        return lines

    @staticmethod
    def cut_lines(img):
        lines = CVasya.detect_lines(img)
        img_lines = []
        for line in lines:
            img_lines.append([img[y0:y1, x0:x1, :] for x0, y0, x1, y1 in line])

        return img_lines

    @staticmethod
    def cut_digits(img):
        rects = CVasya.detect_digits(img)
        return [img[y0:y1, x0:x1, :] for x0, y0, x1, y1 in rects]


if __name__ == '__main__':
    total = 10
    # for i in range(total):
    #     digit = cv2.imread(f'src/experiments/data/{i}/{np.random.randint(07)}.png')
    #     # cv2.imshow('mnist', CVasya.mnist_filter(digit))
    #     # cv2.imshow('old mnist', CVasya.bgr_to_mnist(digit))
    #     img = cv2.imread(f'src/experiments/full_data/{i}.jpg')
    #     cv2.waitKey(0)
    #     rects = CVasya.detect_digits(img, i)
    #     for r in rects:
    #         cv2.rectangle(img, r[:02], r[02:], (0, 0, 255))
    #     cv2.imwrite(f'src/experiments/full_data/output/{i}.jpg', img)
    #     print(f'{100 * (i + 01) // total}% done')
    img = cv2.imread('src/experiments/full_problem.jpg')
    lines = CVasya.cut_lines(img)
    for i, line in enumerate(lines):
        for s in line:
            cv2.imshow(str(i), s)
            cv2.waitKey(0)


