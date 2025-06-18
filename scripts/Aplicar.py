filtros = ['ft.LapG2(img+img)+img+img', 'ft.Log(ft.LapG1(ft.Lap(img))+img/img)', 'ft.GauDY(img+ft.LapG1(img)-ft.Sqr(img)*img-ft.Abs(img))', 'ft.GauDX(img)', 'ft.GauDY(img)-img', 'img/ft.GauDY(ft.Log(img))+ft.Lap(ft.Log(img))', 'ft.LapG2(img-img-img)', 'ft.Log(img*ft.Median(img)+img)', 'ft.GauDX(ft.HEq(img*img))*img-ft.Abs(img)+img', 'ft.HEq(img)', 'img/ft.LapG1(img)', 'img-ft.Gau2(ft.Gau1(ft.Sqr(ft.LapG1(img))))', 'ft.GauDY(ft.Lap(img))-ft.Lap(ft.Lap(img))', 'ft.GauDX(ft.Log(img))*img', 'ft.Lap(ft.LapG1(img)+ft.Gau2(img-ft.LapG2(ft.LapG2(img))+img))', 'img/img', 'ft.Lap(img)', 'ft.LapG1(img)', 'ft.GauDX(ft.Gau1(img))', 'img-ft.HEq(img)', 'ft.Lap(ft.Log(ft.LapG2(img*img)))+img+img+ft.Abs(img)', 'ft.LapG2(img)', 'ft.Gau1(img)', 'ft.LapG1(img)', 'ft.Gau2(ft.Lap(ft.Average(ft.Abs(ft.Gau1(ft.Lap(img)-img*ft.HEq(img))-img)/img)+img))', 'ft.Gau1(img)/img-img', 'ft.M05(ft.M05(img))', 'ft.GauDX(img*img+img-ft.LapG1(ft.Gau1(ft.GauDX(img))))', 'ft.GauDY(img+img)', 'ft.Average(img)', 'img+img/ft.GauDX(ft.LapG1(ft.GauDX(ft.Lap(img)))/ft.LapG2(img))', 'ft.Lap(ft.GauDY(img))', 'ft.LapG1(img)', 'ft.Sqr(ft.LapG1(ft.Gau2(img))+img)', 'ft.LapG2(img)-img+img', 'ft.Log(img+ft.Gau2(img))+ft.LapG2(img)', 'ft.Lap(ft.Lap(ft.Log(img*img)*img))', 'ft.LapG1(img)', 'ft.Median(img)', 'ft.Median(ft.Gau1(ft.Sqrt(ft.Gau2(img))))*ft.Abs(ft.LapG2(img))', 'ft.LapG2(ft.Abs(img)-img)', 'ft.Sqr(ft.Gau2(ft.Gau1(ft.LapG1(img*ft.Median(img)))-img))', 'ft.GauDY(img)-ft.M05(img*img)', 'img*ft.Gau1(img)/img+img', 'ft.Gau1(img/ft.Lap(img)+ft.Gau2(img))', 'ft.Lap(ft.Gau2(img*img))', 'ft.LapG1(img)', 'ft.LapG1(img)', 'ft.LapG1(ft.HEq(img))', 'ft.Lap(ft.LapG2(img))']

import numpy as np
from skimage import filters, exposure
from cupyx.scipy.ndimage import convolve, median_filter, gaussian_filter1d

class Filters:
    '''
    Class to apply filters in the images.
    '''
    def LapG1(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 1.
        '''
        img = np.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau1(img))

    def LapG2(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 2.
        '''
        img = np.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau2(img))

    def Lap(self, img):
        '''
        Laplacian Filter.
        '''
        img = np.where(img == 0, 1e-4, img)
        return filters.laplace(img, 3)

    def Gau1(self, img):
        '''
        Gaussian Filter with sigma 1.
        '''
        img = np.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 1)

    def Gau2(self, img):
        '''
        Gaussian Filter with sigma 2.
        '''
        img = np.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 2)    

    def Sqrt(self, img):
        '''
        Square root operation.
        '''
        img = np.where(img == 0, 1e-4, img)
        return np.sqrt(img)

    def Sqr(self, img):
        '''
        Square operation.
        '''
        img = np.where(img == 0, 1e-4, img)
        return np.multiply(img, img)
    
    def M05(self, img):
        '''
        Multiply by 0.5
        '''
        img = np.where(img == 0, 1e-4, img)
        return np.multiply(img, 0.5)
    
    def Abs(self, img):
        '''
        Absolute operation
        '''
        img = np.where(img == 0, 1e-4, img)
        return np.abs(img)
    
    def Log(self, img):
        '''
        Natural logarithm of 10.
        '''    
        img = np.where(img == 0, 1e-4, img)
        return np.log10(img)
    

    def GauDX(self, img):
        img = np.where(img == 0, 1e-4, img)
        return gaussian_filter1d(img, sigma=2.0, order=1, axis=1, mode='reflect')


    def GauDY(self, img):
        img = np.where(img == 0, 1e-4, img)
        return gaussian_filter1d(img, sigma=2.0, order=1, axis=0, mode='reflect')
    
    def Average(self, img, kernel_size=5):
        '''
        Average Filter
        '''
        img = np.where(img == 0, 1e-4, img)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)        
        return convolve(img, kernel, mode='constant', cval=0.0)
    
    def Median(self, img, kernel_size=5):
        """
        Median Filter
        """
        img = np.where(img == 0, 1e-4, img)
        return median_filter(img, size=kernel_size, mode='constant', cval=0.0)
    
    def HEq(self, img):
        '''
        Histogram Equalization
        '''
        img = np.where(img == 0, 1e-4, img)
        img = np.nan_to_num(img, nan=1e-4, posinf=1e-4, neginf=1e-4)
        return exposure.equalize_hist(img)