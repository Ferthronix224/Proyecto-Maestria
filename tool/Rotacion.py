import cv2

def rotation(image, degree):
    # Ángulo de rotación en grados
    angle_degrees = degree

    # Obtener las dimensiones de la imagen
    height, width = image.shape[:2]

    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Obtener la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

    # Aplicar la rotacion
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # Rellenar con negro

    return rotated_image

rotated_image = rotation(cv2.imread('../img/Imagen5.jpg'), degree=45)
cv2.imwrite('../img/Rotation5.JPG', rotated_image)