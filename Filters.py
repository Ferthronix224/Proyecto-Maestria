import cupy as cp
from cucim.skimage import filters

class Filters:
    '''
    Class to apply filters in the images.
    '''
    def LapG1(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 1
        '''
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau1(img))

    def LapG2(self, img):
        '''
        Laplacian of Gaussian Filter with sigma 2
        '''
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau2(img))

    def Lap(self, img):
        '''
        Laplacian Filter
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.laplace(img, 3)

    def Gau1(self, img):
        '''
        Gaussian Filter with sigma 1
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 1)

    def Gau2(self, img):
        '''
        Gaussian Filter with sigma 2
        '''
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 2)

    def Sqrt(self, img):
        '''
        Square root operation
        '''
        img = cp.where(img == 0, 1e-4, img)
        return cp.sqrt(img)

    def Log(self, img):
            img = cp.where(img == 0, 1e-4, img)
            return cp.log10(img)
    
Filters.LapG1