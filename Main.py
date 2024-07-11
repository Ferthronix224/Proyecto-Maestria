import cv2
import numpy as np
import MP
import DE
from Fitness import Flanned_Matcher

# Ecualizacion
def HEq(img):
    img = cv2.convertScaleAbs(img)
    return cv2.equalizeHist(img)

# Laplaciano-Gaussiano 1
def LapG1(img):
    return Lap(Gau1(img))

# Laplaciano-Gaussiano 2
def LapG2(img):
    return Lap(Gau2(img))

# Laplaciano
def Lap(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    return cv2.Laplacian(img, cv2.CV_64F, ksize=5)

# Gaussiano 1
def Gau1(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)

    # Aplicar el filtro de la Gaussiana
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)

# Gaussiano 2
def Gau2(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)

    # Aplicar el filtro de la Gaussiana
    return cv2.GaussianBlur(img, (5, 5), sigmaX=2, sigmaY=2)

def detectar_puntos_de_interes(magnitud_gradiente, umbral):
    # Crear una máscara booleana donde los elementos mayores que el umbral son True
    mask = magnitud_gradiente > umbral

    # Encontrar los índices de los elementos que son True en la máscara
    indices = np.argwhere(mask)

    return indices


# Función para mostrar puntos de interés
def obtener_puntos_de_interes(imagen, puntos_de_interes, mostrar):
    for c in puntos_de_interes:
        x, y = c.ravel()
        cv2.circle(imagen, center=(x, y), radius=5, color=(0, 0, 255), thickness=-1)
    if mostrar:
        cv2.imshow('imagen', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return imagen

def normalizar(matriz):
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    matriz_normalizada = (matriz - min_val) / (max_val - min_val)
    return matriz_normalizada

def evaluation(img, filter):
    return eval(filter)

# Proceso principal de detección de puntos de interés
def deteccion_de_puntos_de_interes(img1, img2, umbral_deteccion=0.95):
    #  Condicion de si la imagen no se encontro
    if img1 is None or img2 is None:
        raise ValueError("Imagen no encontrada")

    # Cambio de dimensiones
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    # Conversion a escala de grises de la imagen
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Creacion del genotipo por medio de evolucion diferencial
    population = DE.Individual(200, 0, 10).return_genotype()

    # Proceso de mapeo
    filter_MP = MP.generate(population)
    print(filter_MP)

    # Evaluacion de los filtros
    filtro1 = evaluation(img1, filter_MP)
    filtro2 = evaluation(img2, filter_MP)

    # Normalización de los datos
    filtro1_normalizada = normalizar(filtro1)
    filtro2_normalizada = normalizar(filtro2)

    # Detectar puntos de interés basados en la magnitud del gradiente
    puntos_de_interes_1 = detectar_puntos_de_interes(filtro1_normalizada, umbral_deteccion)
    puntos_de_interes_2 = detectar_puntos_de_interes(filtro2_normalizada, umbral_deteccion)

    # Mostrar los puntos de interés detectados
    imagen1 = obtener_puntos_de_interes(img1, puntos_de_interes_1, True)
    imagen2 = obtener_puntos_de_interes(img2, puntos_de_interes_2, True)

    output, repeatability = Flanned_Matcher(imagen1, imagen2)

    # Print the repeatability rate
    print(f'Repeatability rate: {repeatability:.2f}%')

    # Save the image
    cv2.imshow('Match.jpg', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_1 = cv2.imread('img/Diapositiva1.JPG')
    img_2 = cv2.imread('img/Diapositiva2.JPG')
    deteccion_de_puntos_de_interes(img_1, img_2)