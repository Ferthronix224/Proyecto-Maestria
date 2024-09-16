import numpy as np
import cv2

def add_gaussian_noise(image, snr_db):
    """
    Agrega ruido gaussiano a una imagen ajustando el SNR en decibelios.

    :param image: Imagen original
    :param snr_db: Relación señal/ruido en decibelios
    :return: Imagen con ruido gaussiano añadido
    """
    row, col, ch = image.shape
    image = image.astype(np.float32)

    # Calcular la potencia de la señal
    signal_power = np.mean(image ** 2)

    # Calcular la potencia de ruido deseada en base al SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Calcular el ruido gaussiano
    noise = np.random.normal(0, np.sqrt(noise_power), (row, col, ch))

    # Añadir el ruido a la imagen
    noisy_image = image + noise

    # Asegurar que los valores están en el rango adecuado y convertir de nuevo a uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

img = cv2.imread('../img/Imagen5.jpg')
img = cv2.resize(img, (500, 500))
noisy_img = add_gaussian_noise(img, 8)
cv2.imwrite('../img/Noisy5.jpg', noisy_img)
cv2.imshow('img/Noisy.jpg', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()