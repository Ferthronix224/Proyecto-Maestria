import cv2
import numpy as np

def translate(image, pixel_number):
    rows, cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    traslation = np.ones((rows, cols))
    traslation[pixel_number:rows, pixel_number:cols] = image[:rows - pixel_number, :cols - pixel_number]
    traslation = np.uint8(traslation)
    return traslation

for i in range(1, 6):
    translation = translate(cv2.imread(f'../img/Imagen{i}.jpg'), 20)
    cv2.imwrite(f'../img/Translation{i}.jpg', translation)