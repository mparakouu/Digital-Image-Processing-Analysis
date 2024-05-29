import cv2
import numpy as np

football_image = cv2.imread('football.jpg')


pixels = football_image.reshape((-1, 3)).astype(np.float32) # RGB για κάθε pixels


k = 3  # αριθμός των clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) # κριτήρια για τον k-means & αριθμό loops
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)


segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(football_image.shape)


cv2.imshow('football image', football_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()