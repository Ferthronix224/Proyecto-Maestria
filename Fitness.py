# Tasa de repetibilidad
import cv2 as cv

# Función que retorna la imagen con las coincidencias y la tasa de repetibilidad
def Flanned_Matcher(main_image, sub_image):
    # Inicializar el detector SIFT
    sift = cv.SIFT_create()

    # Encontrar puntos de interes y descriptores con SIFT
    key_point1, descr1 = sift.detectAndCompute(main_image, None)
    key_point2, descr2 = sift.detectAndCompute(sub_image, None)

    # Parametros para FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN con implementacion de KNN
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descr1, descr2, k=2)

    # Seleccionar solo buenas coincidencias
    matchesMask = [[0, 0] for i in range(len(matches))]

    good_matches = 0

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.1 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches += 1

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask, flags=0)

    # Dibujar las coincidencias
    img = cv.drawMatchesKnn(main_image, key_point1, sub_image, key_point2, matches, None, **draw_params)

    # Calcular tasa de repetibilidad
    repeatability = good_matches / min(len(key_point1), len(key_point2)) * 100

    return img, repeatability