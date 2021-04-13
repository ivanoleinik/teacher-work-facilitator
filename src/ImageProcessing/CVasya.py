import cv2
import numpy as np
import matplotlib.pyplot as plt


class CVasya:
    """Library for image transforming."""

    @staticmethod
    def bgr_to_mnist(img, interpolation=cv2.INTER_AREA):
        """
        Transform digit image into MNIST format (white digit with black background).

        :param img: Input BGR image.
        :param interpolation: Interpolation type for cv2.resize.
        :return: 28 by 28 greyscale image
        """
        gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray, (28, 28), interpolation=interpolation)
        _, mnist_img = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
        return mnist_img

    @staticmethod
    def mnist_filter(img):
        img_res = CVasya.filter_image_laplacian(img, is_bgr=False)
        kernel = (5, 5)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_DILATE]
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)
        return img_res

    @staticmethod
    def otsu(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)[1]

    @staticmethod
    def filter_image_using_hist(img, fileNum=None):
        # colors, freqs = np.unique(ar=img, axis=0, return_counts=True)
        # sorted_inds = np.flip(np.argsort(freqs))
        # colors, freqs = colors[sorted_inds], freqs[sorted_inds]
        # plt.hist(colors, freqs.reshape(-1, 1), bins='rice')
        plt.hist(img.ravel(), bins='rice')
        if fileNum is not None:
            plt.savefig(f'src/experiments/full_data/output/hist/{fileNum}.jpg')
            plt.close()

    @staticmethod
    def filter_image_laplacian(img, fileNum=None, is_bgr=True):
        c_lap = 1.5
        blur_ker = (9, 9)
        block_lap = 11
        block_agf = 11
        c_agf = 4.5

        if is_bgr:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.blur(gray, blur_ker)
        CVasya.filter_image_using_hist(gray, fileNum)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(f'src/experiments/full_data/output/otsu/{fileNum}.jpg', ~otsu)
        bin_agf = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_agf, c_agf)

        laplace = cv2.Laplacian(gray, cv2.CV_8UC1)
        bin_lap = cv2.adaptiveThreshold(laplace, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_lap, c_lap)

        return ~bin_lap #+ (~bin_agf * 255).astype(np.uint8)) // 2

    @staticmethod
    def detect_digits(img, fileNum=None):
        kernel = (5, 5)
        h, w = img.shape[:2]
        img_res = CVasya.filter_image_laplacian(img, fileNum)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        operations = [cv2.MORPH_CLOSE, cv2.MORPH_DILATE, cv2.MORPH_CLOSE] * 3 # + [cv2.MORPH_DILATE] * 2 + [cv2.MORPH_ERODE]
        for op in operations:
            img_res = cv2.morphologyEx(img_res, op, element)

        cv2.imwrite(f'src/experiments/full_data/output/raw/{i}.jpg', img_res)
        contours, hierarchy = cv2.findContours(img_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = [cv2.boundingRect(c) for c in contours if c.shape[0] > w * h / 1e5]
        rectPoints = [
            (int(r[0] - r[2] * 0.02), int(r[1] - r[3] * 0.08), int(r[0] + r[2] * 1.05), int(r[1] + r[3] * 1.05))
            for r in rects
            if w / 3 > r[2] > w / 100 and h / 3 > r[3] > h / 100
        ]
        return rectPoints


if __name__ == '__main__':
    total = 11
    for i in range(total):
        img = cv2.imread(f'src/experiments/full_data/{i}.jpg')
        rects = CVasya.detect_digits(img, i)
        for r in rects:
            cv2.rectangle(img, r[:2], r[2:], (0, 0, 255))
        cv2.imwrite(f'src/experiments/full_data/output/{i}.jpg', img)
        print(f'{100 * (i + 1) // total}% done')


