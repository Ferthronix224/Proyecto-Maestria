# Tasa de repetibilidad
import cupy as cp

np.set_printoptions(threshold=np.inf)

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
class Fitness:
    def __init__(self, keypoints, transformed_keypoints, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
        self.keypoints = keypoints
        self.transformed_keypoints = transformed_keypoints
        self.low_keypoints_number = low_keypoints_number
        self.up_keypoints_number = up_keypoints_number
        self.transformation = transformation
        self.transformation_value = transformation_value

    def process(self):
        if any([len(self.transformed_keypoints) > self.up_keypoints_number, len(self.keypoints) > self.up_keypoints_number, len(self.transformed_keypoints) < self.low_keypoints_number, len(self.keypoints) < self.low_keypoints_number, len(self.keypoints) == 0, len(self.transformed_keypoints) == 0]):
            return 0

        original_transformed_keypoints = self.transformation(self.keypoints, self.transformation_value)
        distances = cp.linalg.norm(original_transformed_keypoints[:, cp.newaxis] - self.transformed_keypoints, axis=2)
        matches = distances <= 3
        good_matches = cp.sum(cp.any(matches, axis=1))

        # Calcular tasa de repetibilidad
        if good_matches == 0:
            repeatability = 0
        else:
            repeatability = (good_matches / max(len(self.keypoints), len(self.transformed_keypoints))) * 100

        return repeatability