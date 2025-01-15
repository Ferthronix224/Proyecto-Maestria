from skimage import io, transform
import numpy as np

def scale(image, scale_value):
    """
    Escala una imagen y la centra en un lienzo negro con las dimensiones originales.

    Args:
        image (numpy.ndarray): Imagen de entrada.
        scale_value (float): Porcentaje de escala (por ejemplo, 50 para reducir al 50%).

    Returns:
        numpy.ndarray: Imagen escalada y centrada en un lienzo negro.
    """
    # Obtener las dimensiones originales de la imagen
    original_height, original_width = image.shape[:2]

    # Calcular las nuevas dimensiones
    new_width = int(original_width * scale_value / 100)
    new_height = int(original_height * scale_value / 100)

    # Redimensionar la imagen manteniendo la proporción
    resized_image = transform.resize(image, (new_height, new_width), anti_aliasing=True, preserve_range=True)

    # Convertir la imagen redimensionada a uint8 (necesario para manipular en un lienzo negro)
    resized_image = resized_image.astype(np.uint8)

    # Crear un lienzo negro con las dimensiones originales
    output_image = np.zeros((original_height, original_width), dtype=np.uint8)

    # Calcular las coordenadas para centrar la imagen redimensionada
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Colocar la imagen redimensionada en el centro del lienzo negro
    output_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return output_image

# Cargar la imagen
image = io.imread('../img/originals/1.jpg')

# Escalar al 50% y centrar en un lienzo negro
scaled_image = scale(image, 50)

# Guardar la imagen escalada
io.imsave('../img/scale/50/{i}.jpg', scaled_image)
