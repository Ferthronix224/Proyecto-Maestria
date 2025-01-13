import cupy as cp
from cucim.skimage import filters, exposure
from cupyx.scipy.ndimage import convolve, median_filter, gaussian_filter1d

class Filters:
    '''
    Class to apply filters in the images.
    '''
    def LapG1(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 1.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau1(img))

    def LapG2(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 2.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau2(img))

    def Lap(self, img):
        '''
        Laplacian Filter.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.laplace(img, 3)

    def Gau1(self, img):
        '''
        Gaussian Filter with sigma 1.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 1)

    def Gau2(self, img):
        '''
        Gaussian Filter with sigma 2.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 2)    

    def Sqrt(self, img):
        '''
        Square root operation.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return cp.sqrt(img)

    def Sqr(self, img):
        '''
        Square operation.
        '''
        img = cp.where(img == 0, 1e-4, img)
        return cp.multiply(img, img)
    
    def M05(self, img):
        '''
        Multiply by 0.5
        '''
        img = cp.where(img == 0, 1e-4, img)
        return cp.multiply(img, 0.5)
    
    def Abs(self, img):
        '''
        Absolute operation
        '''
        img = cp.where(img == 0, 1e-4, img)
        return cp.abs(img)
    
    def Log(self, img):
        '''
        Natural logarithm of 10.
        '''    
        img = cp.where(img == 0, 1e-4, img)
        return cp.log10(img)
    

    def GauDX(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return gaussian_filter1d(img, sigma=2.0, order=1, axis=1, mode='reflect')


    def GauDY(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return gaussian_filter1d(img, sigma=2.0, order=1, axis=0, mode='reflect')
    
    def Average(self, img, kernel_size=5):
        '''
        Average Filter
        '''
        img = cp.where(img == 0, 1e-4, img)
        kernel = cp.ones((kernel_size, kernel_size), dtype=cp.float32) / (kernel_size ** 2)        
        return convolve(img, kernel, mode='constant', cval=0.0)
    
    def Median(self, img, kernel_size=5):
        """
        Median Filter
        """
        img = cp.where(img == 0, 1e-4, img)
        return median_filter(img, size=kernel_size, mode='constant', cval=0.0)
    
    def HEq(self, img):
        '''
        Histogram Equalization
        '''
        img = cp.where(img == 0, 1e-4, img)
        img = cp.nan_to_num(img, nan=1e-4, posinf=1e-4, neginf=1e-4)
        return exposure.equalize_hist(img)