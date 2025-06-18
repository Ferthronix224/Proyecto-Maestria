from skimage import io, filters, img_as_ubyte, exposure
from scipy.ndimage import convolve, median_filter, gaussian_filter1d
import numpy as np

filtros = ['LapG2(img+img)+img+img', 'Log(LapG1(Lap(img))+img/img)', 'GauDY(img+LapG1(img)-Sqr(img)*img-Abs(img))', 'GauDX(img)', 'GauDY(img)-img', 'img/GauDY(Log(img))+Lap(Log(img))', 'LapG2(img-img-img)', 'Log(img*Median(img)+img)', 'GauDX(HEq(img*img))*img-Abs(img)+img', 'HEq(img)', 'img/LapG1(img)', 'img-Gau2(Gau1(Sqr(LapG1(img))))', 'GauDY(Lap(img))-Lap(Lap(img))', 'GauDX(Log(img))*img', 'Lap(LapG1(img)+Gau2(img-LapG2(LapG2(img))+img))', 'img/img', 'Lap(img)', 'LapG1(img)', 'GauDX(Gau1(img))', 'img-HEq(img)', 'Lap(Log(LapG2(img*img)))+img+img+Abs(img)', 'LapG2(img)', 'Gau1(img)', 'LapG1(img)', 'Gau2(Lap(Average(Abs(Gau1(Lap(img)-img*HEq(img))-img)/img)+img))', 'Gau1(img)/img-img', 'M05(M05(img))', 'GauDX(img*img+img-LapG1(Gau1(GauDX(img))))', 'GauDY(img+img)', 'Average(img)', 'img+img/GauDX(LapG1(GauDX(Lap(img)))/LapG2(img))', 'Lap(GauDY(img))', 'LapG1(img)', 'Sqr(LapG1(Gau2(img))+img)', 'LapG2(img)-img+img', 'Log(img+Gau2(img))+LapG2(img)', 'Lap(Lap(Log(img*img)*img))', 'LapG1(img)', 'Median(img)', 'Median(Gau1(Sqrt(Gau2(img))))*Abs(LapG2(img))', 'LapG2(Abs(img)-img)', 'Sqr(Gau2(Gau1(LapG1(img*Median(img)))-img))', 'GauDY(img)-M05(img*img)', 'img*Gau1(img)/img+img', 'Gau1(img/Lap(img)+Gau2(img))', 'Lap(Gau2(img*img))', 'LapG1(img)', 'LapG1(img)', 'LapG1(HEq(img))', 'Lap(LapG2(img))']

# Cargar imagen y convertir a escala de grises
img = io.imread('img/apple.jpg', as_gray=True)

def LapG1(img):
    '''
    Laplacian of Gaussian Filter with sigma 1.
    '''
    img = np.where(img == 0, 1e-4, img)
    return Lap(Gau1(img))

def LapG2(img):
    '''
    Laplacian of Gaussian Filter with sigma 2.
    '''
    img = np.where(img == 0, 1e-4, img)
    return Lap(Gau2(img))

def Lap(img):
    '''
    Laplacian Filter.
    '''
    img = np.where(img == 0, 1e-4, img)
    return filters.laplace(img, 3)

def Gau1(img):
    '''
    Gaussian Filter with sigma 1.
    '''
    img = np.where(img == 0, 1e-4, img)
    return filters.gaussian(img, 1)

def Gau2(img):
    '''
    Gaussian Filter with sigma 2.
    '''
    img = np.where(img == 0, 1e-4, img)
    return filters.gaussian(img, 2)    

def Sqrt(img):
    '''
    Square root operation.
    '''
    img = np.where(img == 0, 1e-4, img)
    return np.sqrt(img)

def Sqr(img):
    '''
    Square operation.
    '''
    img = np.where(img == 0, 1e-4, img)
    return np.multiply(img, img)

def M05(img):
    '''
    Multiply by 0.5
    '''
    img = np.where(img == 0, 1e-4, img)
    return np.multiply(img, 0.5)

def Abs(img):
    '''
    Absolute operation
    '''
    img = np.where(img == 0, 1e-4, img)
    return np.abs(img)

def Log(img):
    '''
    Natural logarithm of 10.
    '''    
    img = np.where(img == 0, 1e-4, img)
    return np.log10(img)


def GauDX(img):
    img = np.where(img == 0, 1e-4, img)
    return gaussian_filter1d(img, sigma=2.0, order=1, axis=1, mode='reflect')


def GauDY(img):
    img = np.where(img == 0, 1e-4, img)
    return gaussian_filter1d(img, sigma=2.0, order=1, axis=0, mode='reflect')

def Average(img, kernel_size=5):
    '''
    Average Filter
    '''
    img = np.where(img == 0, 1e-4, img)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)        
    return convolve(img, kernel, mode='constant', cval=0.0)

def Median(img, kernel_size=5):
    """
    Median Filter
    """
    img = np.where(img == 0, 1e-4, img)
    return median_filter(img, size=kernel_size, mode='constant', cval=0.0)

def HEq(img):
    '''
    Histogram Equalization
    '''
    img = np.where(img == 0, 1e-4, img)
    img = np.nan_to_num(img, nan=1e-4, posinf=1e-4, neginf=1e-4)
    return exposure.equalize_hist(img)

selected_filter = filtros[np.random.randint(0, len(filtros))]
print(selected_filter)
img_tr = eval(selected_filter)

# Normalizar y convertir a uint8 para guardar correctamente
img_lap_uint8 = img_as_ubyte((img_tr - np.min(img_tr)) / (np.max(img_tr) - np.min(img_tr)))

print(img_lap_uint8.max(), img_lap_uint8.min())
# Guardar imagen
io.imsave('img/apple_tr.jpg', img_lap_uint8)
