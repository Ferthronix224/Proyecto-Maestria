from skimage import io, transform, img_as_ubyte
import numpy as np

def scale(image, scale_value):
    """
    Escala una imagen y la centra en un lienzo negro con las dimensiones originales.
    
    Args:
        image (numpy.ndarray): Imagen de entrada (grayscale o RGB).
        scale_value (float): Porcentaje de escala (por ejemplo, 80 para reducir al 80%).
    
    Returns:
        numpy.ndarray: Imagen escalada y centrada en un lienzo negro.
    """
    original_height, original_width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    # Nuevas dimensiones
    new_width = int(original_width * scale_value / 100)
    new_height = int(original_height * scale_value / 100)

    # Redimensionar imagen
    resized_image = transform.resize(image, (new_height, new_width), anti_aliasing=True, preserve_range=True)
    resized_image = resized_image.astype(np.uint8)

    # Lienzo negro (2D o 3D)
    if channels == 1:
        output_image = np.zeros((original_height, original_width), dtype=np.uint8)
    else:
        output_image = np.zeros((original_height, original_width, channels), dtype=np.uint8)

    # Coordenadas de centrado
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Colocar imagen centrada
    if channels == 1:
        output_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    else:
        output_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_image

    return output_image

image = io.imread('img/figures.png')  # sin `as_gray=True` para soportar RGB
scaled_image = scale(image, 70)
io.imsave('img/figures_scaled.png', img_as_ubyte(scaled_image))
