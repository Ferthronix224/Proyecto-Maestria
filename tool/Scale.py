import cv2
import numpy as np

def scale(image, scale_value):
    # Obtener las dimensiones originales de la imagen
    original_height, original_width = image.shape[:2]

    new_width = int(original_width * scale_value / 100)
    new_height = int(original_height * scale_value / 100)

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

for i in range(1, 6):
    values = [90, 80, 70, 60, 50]
    image = scale(cv2.imread(f'../img/img{i}.jpg'), values[i - 1])
    cv2.imwrite(f'../img/Scale{i}.jpg', image)
