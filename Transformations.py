import numpy as np

def translate(coor, pixel):
    return coor + pixel

def rotation(coordenadas, angulo_grados):
    """
    Rota una lista de puntos 2D alrededor de un centro dado en sentido antihorario.

    :param coordenadas: Lista bidimensional de coordenadas [[x1, y1], [x2, y2], ...]
    :param angulo_grados: Ángulo de rotación en grados (antihorario)
    :param centro: Centro de rotación, por defecto es el origen (0, 0)
    :return: Lista de coordenadas transformadas
    """
    # Convertir el ángulo a radianes
    angulo_radianes = np.radians(angulo_grados)
    
    # Matriz de rotación
    R = np.array([[np.cos(angulo_radianes), -np.sin(angulo_radianes)],
                  [np.sin(angulo_radianes), np.cos(angulo_radianes)]])
    
    # Desplazamiento del centro si no es el origen
    centro = np.array((128, 128))
    
    # Aplicar la rotación a cada punto
    nuevas_coordenadas = []
    for coord in coordenadas:
        # Trasladar el punto al origen según el centro
        punto_trasladado = np.array(coord) - centro
        # Rotar el punto
        punto_rotado = R.dot(punto_trasladado)
        # Desplazar el punto de vuelta
        nueva_coordenada = punto_rotado + centro
        nuevas_coordenadas.append(nueva_coordenada.tolist())
    
    return np.array(nuevas_coordenadas)

def scale(coordenadas, escala):
    """
    Escala una matriz de coordenadas en torno a un centro (cx, cy) y factores de escala sx, sy.
    
    matriz: np.array de tamaño (N, 2), donde N es el número de puntos, con columnas (x, y).
    escala: Factores de escala 
    """
    coordenadas = coordenadas.astype(float)
    # Crear una copia de la matriz para no modificar la original
    matriz_escalada = np.copy(coordenadas)
    
    # Restar el centro de escala
    matriz_escalada[:, 0] -= 128
    matriz_escalada[:, 1] -= 128

    # Aplicar el factor de escala a las coordenadas x e y
    matriz_escalada[:, :] *= escala / 100

    # Volver a trasladar al sistema original sumando el centro de escala
    matriz_escalada[:, 0] += 128
    matriz_escalada[:, 1] += 128

    return np.array(matriz_escalada.astype(int))