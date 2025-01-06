import cupy as cp
import gc
from Filters import Filters
from MP import MP
from Fitness import Fitness

class Process:
  def detectar_puntos_de_interes(self, magnitud, umbral):
    mask = cp.asarray(magnitud) > umbral
    indices = cp.argwhere(mask)
    return cp.asarray(indices)

  def normalizar(self, matriz):
      matriz = cp.where(cp.isnan(matriz), 1e-4, matriz)
      min_val = cp.min(matriz)
      max_val = cp.max(matriz)
      if any([max_val == 0.0, min_val == -cp.inf, max_val == cp.inf]):
          return cp.zeros_like(matriz)
      matriz_normalizada = cp.divide(cp.subtract(matriz, min_val), cp.subtract(max_val, min_val))
      return matriz_normalizada

  def evaluation(self, img, filter_str):
    ft = Filters()
    if filter_str == 'Worst':
        return cp.zeros_like(img)
    return eval(filter_str)

  def apply_filters(self, images, filters):
    results = []
    for filter_str in filters:
        # Evaluar cada filtro para todas las imágenes
        filtered_imgs = cp.asarray([self.evaluation(img, filter_str) for img in images])
        results.append(filtered_imgs)
    return cp.asarray(results)

  def repeatability(self, population, img1, img2, umbral_deteccion, wr, low_keypoints_number, up_keypoints_number, transformation, transformation_value):
      # Proceso de mapeo
      filter_MP = [MP().generate(population[i], wr) for i in range(len(population))]
      img_len = len(img1)

      # Evaluacion de los filtros
      filtro1 = self.apply_filters(img1, filter_MP).astype(cp.uint8)
      filtro2 = self.apply_filters(img2, filter_MP).astype(cp.uint8)
      del img1, img2
      cp._default_memory_pool.free_all_blocks()
      gc.collect()

      # Normalización de los datos
      filtro1_normalizado = [[self.normalizar(filtro1[i][ii]) for ii in range(len(filtro1[0]))] for i in range(len(filtro1))]
      del filtro1
      filtro2_normalizado = [[self.normalizar(filtro2[i][ii]) for ii in range(len(filtro2[0]))] for i in range(len(filtro2))]
      del filtro2
      cp._default_memory_pool.free_all_blocks()
      gc.collect()

      # Puntos de interes
      puntos_de_interes_1 = [[self.detectar_puntos_de_interes(filtro1_normalizado[i][ii], umbral_deteccion) for ii in range(len(filtro1_normalizado[0]))] for i in range(len(filtro1_normalizado))]
      del filtro1_normalizado

      puntos_de_interes_2 = [[self.detectar_puntos_de_interes(filtro2_normalizado[i][ii], umbral_deteccion) for ii in range(len(filtro2_normalizado[0]))] for i in range(len(filtro2_normalizado))]
      del filtro2_normalizado
      cp._default_memory_pool.free_all_blocks()
      gc.collect()

      # Repeatability
      repeatability = [[Fitness(puntos_de_interes_1[i][ii], puntos_de_interes_2[i][ii], low_keypoints_number, up_keypoints_number, transformation, transformation_value).process() for ii in range(img_len)] for i in range(len(filter_MP))]
      del puntos_de_interes_1, puntos_de_interes_2
      cp._default_memory_pool.free_all_blocks()
      gc.collect()

      return [sum(row) / len(row) for row in repeatability], filter_MP