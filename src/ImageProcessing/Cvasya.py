import cv2


class Cvasya:
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
    def otsu(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)[1]
        
    @staticmethod
    def detect_digits_bgr(img):
        c_lap = 1.9
        c_agf = 5
        blur_ker = (9, 9)
        block_lap = 11
        block_agf = 11

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, blur_ker)
        bin_agf = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_agf, c_agf)
        laplace = cv2.Laplacian(gray, cv2.CV_8U)
        bin_lap = cv2.adaptiveThreshold(laplace, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_lap, c_lap)

        contours_lap, _ = cv2.findContours(
                                      image=bin_lap,
                                      mode=cv2.RETR_TREE,
                                      method=cv2.CHAIN_APPROX_NONE
                          )
        contours_agf, _ = cv2.findContours(
                                      image=bin_agf,
                                      mode=cv2.RETR_TREE,
                                      method=cv2.CHAIN_APPROX_NONE
                          )
        img_lap = img.copy()
        img_agf = img.copy()
        cv2.drawContours(
            image=img_lap,
            contours=contours_lap,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.drawContours(
            image=img_agf,
            contours=contours_agf,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.imshow('Adaptive gaussian filter', img_agf)
        cv2.imshow('Laplacian', img_lap)
        cv2.waitKey(0)


paths = [f'experiments/full_data/{i}.jpg' for i in range(8)]
for path in paths:
    Cvasya.detect_digits_bgr(cv2.imread(path))
