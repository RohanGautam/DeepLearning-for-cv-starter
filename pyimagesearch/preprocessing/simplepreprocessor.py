import cv2


class SimplePreprocessor:
    def __init__(self, width, height, interp=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interp = interp

    def preprocess(self, image):
        # resize ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.interp)
