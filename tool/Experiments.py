import numpy as np

lista = np.array([[2,2,4,5,0], [0,0,0,0,0], [0,2,3,5,6]])
lista = np.where(lista == 0, 24, lista)
print(lista)