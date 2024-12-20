import cupy as cp
from skimage import filters

# Laplaciano-Gaussiano 1
class Filters:
    def LapG1(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau1(img))

    def LapG2(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return self.Lap(self.Gau2(img))

    def Lap(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return filters.laplace(img, 3)

    def Gau1(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 1)

    def Gau2(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return filters.gaussian(img, 2)

    def Sqrt(self, img):
        img = cp.where(img == 0, 1e-4, img)
        return cp.sqrt(img)

    def Log(self, img):
            img = cp.where(img == 0, 1e-4, img)
            return cp.log10(img)