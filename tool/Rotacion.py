import cv2

image = cv2.imread('../img/Cuadrado 3.JPG')

# Ángulo de rotación en grados
angle_degrees = 15

# Obtener las dimensiones de la imagen
height, width = image.shape[:2]

# Calcular el centro de la imagen
center = (width // 2, height // 2)

# Obtener la matriz de rotación
rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

# Aplicar la rotación
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Guardar imagen rotada
cv2.imwrite('../img/Rotation.JPG', rotated_image)