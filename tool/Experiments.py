import Transformations as tr
import cv2

ruido = tr.noise
noisy_img = ruido(cv2.imread('../img/Imagen1.jpg'), 8)
cv2.imwrite('Noisy.jpg', noisy_img)