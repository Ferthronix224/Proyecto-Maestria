import numpy as np
import time

# inicio = time.time()

indices = np.array([[1, np.random.randint(1, 100)] for _ in range(10000)])
rotated_keypoints = np.array([[2, np.random.randint(1, 100)] for _ in range(10000)])

print(len(indices), indices.shape, indices.size)

# distances = np.linalg.norm(indices[:, np.newaxis] - rotated_keypoints, axis=2)
# good_matches = np.sum(distances <= 5)
# print(good_matches)
#
# fin = time.time()
#
# print(fin - inicio)