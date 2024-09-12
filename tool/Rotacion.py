import cv2

image = cv2.imread('../img/3317.jpg')

# Ángulo de rotación en grados
angle_degrees = 5

# Obtener las dimensiones de la imagen
height, width = image.shape[:2]

# Calcular el centro de la imagen
center = (width // 2, height // 2)

# Obtener la matriz de rotación
rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

# Aplicar la rotación
border_color = (255, 255, 255)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)  # Rellenar con blanco

# Guardar imagen rotada
cv2.imwrite('../img/Rotation.JPG', rotated_image)