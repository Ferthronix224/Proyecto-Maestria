import numpy as np
import cv2

# Ecualizacion
def HEq(img):
    img = np.uint8(img)
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

# Raíz Cuadrada
def Sqrt(input_list_2d):
    result = []
    for sublist in input_list_2d:
        result_sublist = []
        for x in sublist:
            if x < 0:
                result_sublist.append(x)
            else:
                result_sublist.append(np.sqrt(x))
        result.append(result_sublist)
    return np.array(result)

# Logaritmo
def Log(input_list_2d):
    result = []
    for sublist in input_list_2d:
        result_sublist = []
        for x in sublist:
            if x <= 0:
                result_sublist.append(x)
            else:
                result_sublist.append(np.sqrt(x))
        result.append(result_sublist)
    return np.array(result)