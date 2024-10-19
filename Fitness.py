# Tasa de repetibilidad
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
class Fitness:
    def __init__(self, image, transformed_image, keypoints, transformed_keypoints, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
        self.image = image
        self.transformed_image = transformed_image
        self.keypoints = keypoints
        self.transformed_keypoints = transformed_keypoints
        self.low_keypoints_number = low_keypoints_number
        self.up_keypoints_number = up_keypoints_number
        self.transformation = transformation
        self.transformation_value = transformation_value

    def process(self):
        if len(self.transformed_keypoints) > self.up_keypoints_number or len(self.keypoints) > self.up_keypoints_number or len(self.transformed_keypoints) < self.low_keypoints_number or len(self.keypoints) < self.low_keypoints_number or len(self.keypoints) == 0 or len(self.transformed_keypoints) == 0 or type(self.image) is int or type(self.transformed_image) is int:
            return 0, 0, 0
        
        original_transformed_keypoints = self.transformation(self.keypoints, self.transformation_value)
        distances = np.linalg.norm(original_transformed_keypoints[:, np.newaxis] - self.transformed_keypoints, axis=2)
        matches = distances <= 3
        good_matches = 0

        self.image = np.uint8(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.transformed_image = np.uint8(self.transformed_image)
        self.transformed_image = cv2.cvtColor(self.transformed_image, cv2.COLOR_GRAY2BGR)

        for i, match in enumerate(matches):
            if True in match:
                good_matches += 1
                x, y = self.keypoints[i].ravel().astype(int)
                self.image = cv2.circle(self.image, center=(y, x), radius=1, color=(0, 255, 0), thickness=-1, )
            else:
                x, y = self.keypoints[i].ravel().astype(int)
                self.image = cv2.circle(self.image, center=(y, x), radius=1, color=(0, 0, 255), thickness=-1)

        for i, match in enumerate(matches.transpose()):
            if True in match:
                x, y = self.transformed_keypoints[i].ravel().astype(int)
                self.transformed_image = cv2.circle(self.transformed_image, center=(y, x), radius=1, color=(0, 255, 0), thickness=-1)
            else:
                x, y = self.transformed_keypoints[i].ravel().astype(int)
                self.transformed_image = cv2.circle(self.transformed_image, center=(y, x), radius=1, color=(0, 0, 255), thickness=-1)

        # Calcular tasa de repetibilidad
        if good_matches == 0:
            repeatability = 0
        else:
            repeatability = (good_matches / max(len(self.keypoints), len(self.transformed_keypoints))) * 100

        return repeatability, self.image, self.transformed_image