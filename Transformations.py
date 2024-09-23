import cv2
import numpy as np

def translate(image, pixel_number):
    rows, cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    traslation = np.ones((rows, cols))
    traslation[pixel_number:rows, pixel_number:cols] = image[:rows - pixel_number, :cols - pixel_number]
    traslation = np.uint8(traslation)
    return traslation
def scale(image, scale_value):
    # Obtener las dimensiones originales de la imagen
    original_height, original_width = image.shape[:2]

    # Definir las nuevas dimensiones deseadas (por ejemplo, reducir el tamaño)
    scale_percent = scale_value  # Escalar al 50%
    new_width = int(original_width * scale_percent)
    new_height = int(original_height * scale_percent)

    # Redimensionar la imagen manteniendo la proporción
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Crear una nueva imagen (lienzo) con las dimensiones originales y fondo negro
    output_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Calcular las coordenadas de la esquina superior izquierda para centrar la imagen redimensionada
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Colocar la imagen redimensionada en el centro del lienzo negro
    output_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    return output_image
def rotation(image, degree):
    # Obtener las dimensiones de la imagen
    height, width = image.shape[:2]

    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Obtener la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, degree, scale=1.0)

    # Conversion a float32
    image = np.float32(image)

    # Aplicar la rotacion
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # Rellenar con negro

    return np.uint8(rotated_image)

def noise(image, snr_db):
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