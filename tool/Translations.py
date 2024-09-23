import cv2
import numpy as np

def translate(image, pixel_number):
    rows, cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    traslation = np.ones((rows, cols))
    traslation[pixel_number:rows, pixel_number:cols] = image[:rows - pixel_number, :cols - pixel_number]
    traslation = np.uint8(traslation)
    return traslation

translation = translate(cv2.imread('../img/Imagen1.jpg'), 50)
cv2.imwrite('Translation.jpg', translation)