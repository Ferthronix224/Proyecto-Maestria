import cupy as cp

class Fitness:
    '''
    Class to measure phenotypes' performance.

    Parameters
    ----------
    interest_points (array): Array of interest points coordinates from the original image.
    transformed_interest_points (array): Array of interest points coordinates from the transformed image.
    low_interest_points_number (int): Low number range of interest points.
    up_interest_points_number (int): Up number range of interest points.
    transformation (function): Function to transform interest points into transformed interest points.
    transformation_value (int): Transformation value to use in the transformation function.
    '''
    # Constructor.
    def __init__(self, interest_points, transformed_interest_points, low_interest_points_number, up_interest_points_number, transformation, transformation_value):
        self.interest_points = interest_points
        self.transformed_interest_points = transformed_interest_points
        self.low_interest_points_number = low_interest_points_number
        self.up_interest_points_number = up_interest_points_number
        self.transformation = transformation
        self.transformation_value = transformation_value

    def process(self):
        # Condition to prevent some variables from not being in the established range.
        if any([len(self.transformed_interest_points) > self.up_interest_points_number, len(self.interest_points) > self.up_interest_points_number, len(self.transformed_interest_points) < self.low_interest_points_number, len(self.interest_points) < self.low_interest_points_number, len(self.interest_points) == 0, len(self.transformed_interest_points) == 0]):
            return 0

        # Interest points' matching.
        original_transformed_interest_points = self.transformation(self.interest_points, self.transformation_value)
        # Distance calculation.
        distances = cp.linalg.norm(original_transformed_interest_points[:, cp.newaxis] - self.transformed_interest_points, axis=2)
        del original_transformed_interest_points
        # Mask
        matches = distances <= 1.5
        del distances
        # Good matches calculation.
        good_matches = cp.sum(cp.any(matches, axis=1))
        del matches

        # Calculate repeatability rate.
        if good_matches == 0:
            repeatability = 0
        else:
            repeatability = (good_matches / max(len(self.interest_points), len(self.transformed_interest_points))) * 100
            
        del good_matches

        return repeatability