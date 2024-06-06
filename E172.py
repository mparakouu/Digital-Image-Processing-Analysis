import numpy as np
import cv2
from sklearn.cluster import KMeans

football_image = cv2.imread('football.jpg')
football_image = cv2.cvtColor(football_image, cv2.COLOR_BGR2RGB)


                    # K-MEANS ALGORITHM
pixelVal = football_image.reshape((-1, 3))
k_values = [2, 3, 4]

for k in k_values:
    kmeans = KMeans(n_clusters=k) # αριθμός των cluster
    kmeans.fit(pixelVal) # pixel --> k clusters

    labels = kmeans.labels_

    for i in range(k):
        # pixels καθε ομάδας --> football_cluster_image
        cluster_indices = np.where(labels == i)[0]
        football_cluster_image = np.zeros_like(football_image) # πίνακας 
        for pixel in cluster_indices:
            x, y = np.unravel_index(pixel, football_image.shape[:2]) # pixel --> x , y
            football_cluster_image[x, y] = football_image[x, y]

        cv2.imshow(f'k={k}, cluster={i}', football_cluster_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



