from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt

# %%
dev = True


def showImage(img):
    '''Expects a numpy array (how cv2 reads it)'''
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # to convert BGR to RGB
    plt.title('Image')
    plt.show()


image = cv2.imread("datasets/dog.png")

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])
# %%

# get height and width from matrix dimensions.
imH, imW, _ = image.shape
krH, krW = kernel.shape
# Check if kernel is a square matrix
assert krH == krW

# %%
# Step1: Padd the borders so that we can use the kernel at the edges too.
padBy = krH//2  # or (krH-1)/2
# pad by replicating current border
image = cv2.copyMakeBorder(image, padBy, padBy, padBy, padBy,
                           cv2.BORDER_REPLICATE)
if dev:
    showImage(image)

# %%
# Step2: 'slide' over the matrix
# initialize output with zeroes
output = np.zeros((imH, imW), dtype="float")
for x in range(padBy, imW+padBy):
    for y in range(padBy, imH+padBy):
        pixel = (x, y)
        branchLen = krH//2
        # the values will never be negative as we have padded the image.
        imageSlice = image[pixel[1]-branchLen: pixel[1] +
                           branchLen+1, pixel[0]-branchLen: pixel[0]+branchLen+1]

        output[y-padBy, x-padBy] = (imageSlice*kernel).sum()

# rescale the output image to be in the range [0, 255]
output = rescale_intensity(output, in_range=(0, 255))
output = (output * 255).astype("uint8")
showImage(output)


# %%
# To wrap it up:
def convolute(image, kernel):
    imH, imW, _ = image.shape
    krH, krW = kernel.shape
    # Check if kernel is a square matrix
    assert krH == krW
    padBy = (krH-1)//2  # or (krH-1)/2
    # pad by replicating current border
    image = cv2.copyMakeBorder(image, padBy, padBy, padBy, padBy,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((imH, imW), dtype="float")
    for x in range(padBy, imW+padBy):
        for y in range(padBy, imH+padBy):
            pixel = (x, y)
            branchLen = padBy
            # the values will never be negative as we have padded the image.
            imageSlice = image[y-branchLen: y +
                               branchLen+1, x-branchLen: x+branchLen+1]

            output[y-padBy, x-padBy] = (imageSlice*kernel).sum()

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    showImage(output)

# %%


blur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
blur = np.ones((3, 3), dtype="float") * (1.0 / (3*3))
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
# edge-like regions
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")
convolute(image, blur)

# %%
