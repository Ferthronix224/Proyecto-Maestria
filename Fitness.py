import cupy as cp

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
        del original_transformed_keypoints
        matches = distances <= 3
        del distances
        good_matches = cp.sum(cp.any(matches, axis=1))
        del matches

        # Calcular tasa de repetibilidad
        if good_matches == 0:
            repeatability = 0
        else:
            repeatability = (good_matches / max(len(self.keypoints), len(self.transformed_keypoints))) * 100
            
        del good_matches

        return repeatability