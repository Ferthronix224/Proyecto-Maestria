from skimage import io
import numpy as np

def translate(image, pixel_number):
    rows, cols = image.shape
    traslation = np.ones((rows, cols))
    traslation[pixel_number:rows, pixel_number:cols] = image[:rows - pixel_number, :cols - pixel_number]
    traslation = np.uint8(traslation)
    return traslation

for i in range(1, 293):
    translation = translate(io.imread(f'../img/originals/{i}.jpg'), 20)
    io.imsave(f'../img/translated/20/{i}.jpg', translation)