import cv2
import numpy as np

A = cv2.imread('../img/Imagen5.jpg')
A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
B = np.ones((256, 256))
B[19:255, 19:255] = A[:236, :236]
B = np.uint8(B)
cv2.imwrite('../img/Translation5.jpg', B)