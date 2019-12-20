import numpy as np
import cv2
import os


class SimpleDatasetloader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors  # we might have to apply many of them
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            # os.path.sep is '/', this gets the class
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    # apply the preprocessor to the image
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
