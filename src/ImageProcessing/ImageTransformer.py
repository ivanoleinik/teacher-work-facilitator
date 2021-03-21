import cv2


class ImageTransformer:
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
