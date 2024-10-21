import numpy as np
import cv2

# Laplaciano-Gaussiano 1
def LapG1(img):
    img = np.where(img == 0, 1e-4, img)
    return Lap(Gau1(img.copy()))

# Laplaciano-Gaussiano 2
def LapG2(img):
    img = np.where(img == 0, 1e-4, img)
    return Lap(Gau2(img.copy()))

# Laplaciano
def Lap(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    img = np.where(img == 0, 1e-4, img)
    return cv2.Laplacian(img, cv2.CV_64F, ksize=5)

# Gaussiano 1
def Gau1(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    img = np.where(img == 0, 1e-4, img)
    # Aplicar el filtro de la Gaussiana
    # print('Gau1')
    value = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)
    # print(value)
    return value

# Gaussiano 2
def Gau2(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    img = np.where(img == 0, 1e-4, img)
    # Aplicar el filtro de la Gaussiana
    return cv2.GaussianBlur(img, (5, 5), sigmaX=2, sigmaY=2)

# Raíz Cuadrada
def Sqrt(img):
    img = np.where(img == 0, 1e-4, img)
    result = []
    for sublist in img:
        result_sublist = []
        for x in sublist:
            if x < 0:
                result_sublist.append(x)
            else:
                result_sublist.append(np.sqrt(x))
        result.append(result_sublist)
    return np.array(result)

# Logaritmo
def Log(img):
    img = np.where(img == 0, 1e-4, img)
    result = []
    for sublist in img:
        result_sublist = []
        for x in sublist:
            if x <= 0:
                result_sublist.append(x)
            else:
                result_sublist.append(np.log10(x))
        result.append(result_sublist)
    return np.array(result)