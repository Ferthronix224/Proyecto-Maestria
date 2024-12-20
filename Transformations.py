import cupy as cp

class Transformations:
  def translate(self, coor, pixel):
    return coor + pixel

  def rotation(self, coordenadas, angulo_grados):
      """
      Rota una lista de puntos 2D alrededor de un centro dado en sentido antihorario.

      :param coordenadas: Lista bidimensional de coordenadas [[x1, y1], [x2, y2], ...]
      :param angulo_grados: Ángulo de rotación en grados (antihorario)
      :param centro: Centro de rotación, por defecto es el origen (0, 0)
      :return: Lista de coordenadas transformadas
      """
      # Convertir el ángulo a radianes
      angulo_radianes = cp.radians(angulo_grados)

      # Matriz de rotación
      R = cp.array([[cp.cos(angulo_radianes), -cp.sin(angulo_radianes)],
                    [cp.sin(angulo_radianes), cp.cos(angulo_radianes)]])

      # Desplazamiento del centro (puedes parametrizarlo si lo necesitas)
      centro = cp.array([128, 128])

      # Convertir coordenadas a cupy array si no lo son
      coordenadas = cp.array(coordenadas)

      # Trasladar todos los puntos al origen
      coordenadas_trasladadas = coordenadas - centro

      # Aplicar la rotación vectorizada
      coordenadas_rotadas = coordenadas_trasladadas @ R.T

      # Trasladar los puntos de vuelta
      nuevas_coordenadas = coordenadas_rotadas + centro

      return nuevas_coordenadas

  def scale(self, coordenadas, escala):
      """
      Escala una matriz de coordenadas en torno a un centro (cx, cy) y factores de escala sx, sy.

      matriz: cp.array de tamaño (N, 2), donde N es el número de puntos, con columnas (x, y).
      escala: Factores de escala
      """
      coordenadas = coordenadas.astype(float)
      # Crear una copia de la matriz para no modificar la original
      matriz_escalada = cp.copy(coordenadas)

      # Restar el centro de escala
      matriz_escalada[:, 0] -= 128
      matriz_escalada[:, 1] -= 128

      # Aplicar el factor de escala a las coordenadas x e y
      matriz_escalada[:, :] *= escala / 100

      # Volver a trasladar al sistema original sumando el centro de escala
      matriz_escalada[:, 0] += 128
      matriz_escalada[:, 1] += 128

      return cp.array(matriz_escalada.astype(int))