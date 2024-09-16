import cv2

image = cv2.imread('../img/Imagen5.jpg')

# Ángulo de rotación en grados
angle_degrees = 5

# Obtener las dimensiones de la imagen
height, width = image.shape[:2]

# Calcular el centro de la imagen
center = (width // 2, height // 2)

# Obtener la matriz de rotación
rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

# Aplicar la rotación
border_color = (0, 0, 0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)  # Rellenar con blanco

# Guardar imagen rotada
cv2.imwrite('../img/Rotation5.JPG', rotated_image)