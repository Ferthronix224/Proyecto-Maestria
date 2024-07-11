# Repeatability Rate
import cv2 as cv

# Function returns the feature matched image and repeatability rate
def Flanned_Matcher(main_image, sub_image):
    # Initiating the SIFT detector
    sift = cv.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    key_point1, descr1 = sift.detectAndCompute(main_image, None)
    key_point2, descr2 = sift.detectAndCompute(sub_image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN based matcher with implementation of k nearest neighbour
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descr1, descr2, k=2)

    # Selecting only good matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    good_matches = 0

    # Ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.1 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches += 1

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask, flags=0)

    # Drawing nearest neighbours
    img = cv.drawMatchesKnn(main_image, key_point1, sub_image, key_point2, matches, None, **draw_params)

    # Calculate repeatability rate
    repeatability = good_matches / min(len(key_point1), len(key_point2)) * 100

    return img, repeatability

if __name__ == '__main__':
    # Reading two input images
    main_image = cv.imread('img/Diapositiva1.JPG')
    sub_image = cv.imread('Noisy.jpg')

    # Passing two input images
    output, repeatability = Flanned_Matcher(main_image, sub_image)

    # Print the repeatability rate
    print(f'Repeatability rate: {repeatability:.2f}%')

    # Save the image
    cv.imshow('Match.jpg', output)
    cv.waitKey(0)
    cv.destroyAllWindows()