import numpy as np

arrar = np.array([[5, 5, 6], [2, 4, 6], [2, 4, 6], [7, 7, 6]])

MASK = (arrar <= 5)

good_matches = 0

for i in MASK:
    if True in i:
        good_matches +=1

print(good_matches)