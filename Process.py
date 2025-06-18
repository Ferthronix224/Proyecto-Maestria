# Libraries
import numpy as cp
import gc
# Scripts
from Filters import Filters
from MP import MP
from Fitness import Fitness

class Process:    
    '''
    Class to process images.
    '''
    # Function to obtain interest point coordinates.
    def interest_points_detection(self, magnitud, umbral):
        mask = cp.asarray(magnitud) > umbral
        indices = cp.argwhere(mask)
        return cp.asarray(indices)

    # Function to convert the image's range into 0 to 1.
    def normalization(self, matriz):
        matriz = cp.where(cp.isnan(matriz), 1e-4, matriz)
        min_val = cp.min(matriz)
        max_val = cp.max(matriz)
        if any([max_val == 0.0, min_val == -cp.inf, max_val == cp.inf]):
            return cp.zeros_like(matriz)
        matriz_normalizada = cp.divide(cp.subtract(matriz, min_val), cp.subtract(max_val, min_val))
        return matriz_normalizada

    # Function to apply a filter to an image.
    def evaluation(self, img, filter_str):
        ft = Filters()
        if filter_str == 'Worst':
            return cp.zeros_like(img)
        return eval(filter_str)

    # Function to evaluate each filter for all the images.
    def apply_filters(self, images, filters):
        results = []
        for filter_str in filters:
            # Evaluar cada filtro para todas las imágenes.
            filtered_imgs = cp.array([self.evaluation(img, filter_str) for img in images])
            results.append(filtered_imgs)
        return cp.array(results)

    # Function to return repeatability population and its corresponding filters.
    def repeatability(self, population, img1, img2, umbral_deteccion, wr, low_interest_points_number, up_interest_points_number, transformation, transformation_value):
        # Mapping Process.
        filter_MP = [MP().generate(population[i], wr) for i in range(len(population))]
        img_len = len(img1)

        # Filters evaluation.
        filter_1 = self.apply_filters(img1, filter_MP).astype(cp.uint8)
        filter_2 = self.apply_filters(img2, filter_MP).astype(cp.uint8)
        del img1, img2

        # Normalization.
        normalized_filter_1 = [[self.normalization(filter_1[i][ii]) for ii in range(len(filter_1[0]))] for i in range(len(filter_1))]
        del filter_1
        normalized_filter_2 = [[self.normalization(filter_2[i][ii]) for ii in range(len(filter_2[0]))] for i in range(len(filter_2))]
        del filter_2

        # Interest Points.
        interest_point_1 = [[self.interest_points_detection(normalized_filter_1[i][ii], umbral_deteccion) for ii in range(len(normalized_filter_1[0]))] for i in range(len(normalized_filter_1))]
        del normalized_filter_1
        interest_point_2 = [[self.interest_points_detection(normalized_filter_2[i][ii], umbral_deteccion) for ii in range(len(normalized_filter_2[0]))] for i in range(len(normalized_filter_2))]
        del normalized_filter_2

        # Repeatability.
        repeatability = [[Fitness(interest_point_1[i][ii], interest_point_2[i][ii], low_interest_points_number, up_interest_points_number, transformation, transformation_value).process() for ii in range(img_len)] for i in range(len(filter_MP))]
        del interest_point_1, interest_point_2

        return [sum(row) / len(row) for row in repeatability], filter_MP