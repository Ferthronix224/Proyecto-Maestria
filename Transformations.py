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
      
      del angulo_radianes

      centro = cp.array([128, 128])

      # Trasladar todos los puntos al origen
      coordenadas -= centro

      # Aplicar la rotación vectorizada
      coordenadas @= R.T
      
      del R

      # Trasladar los puntos de vuelta
      coordenadas += centro

      del centro

      return coordenadas

  def scale(self, coordenadas, escala):
      """
      Escala una matriz de coordenadas en torno a un centro (cx, cy) y factores de escala sx, sy.

      matriz: cp.array de tamaño (N, 2), donde N es el número de puntos, con columnas (x, y).
      escala: Factores de escala
      """
      coordenadas = coordenadas.astype(float)
      centro = cp.array([128, 128])
      # Restar el centro de escala
      coordenadas -= centro

      # Aplicar el factor de escala a las coordenadas x e y
      coordenadas *= escala / 100

      # Volver a trasladar al sistema original sumando el centro de escala
      coordenadas += centro

      del centro

      return cp.array(coordenadas.astype(int))