import cv2
import numpy as np

# football_image
football_image = cv2.imread('football.jpg')

# 2D pixel array και --> float 32
pixels = football_image.reshape((-1, 3)).astype(np.float32)

# k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# k-means --> k = 2 , 3 , 4
k_values = [2, 3, 4]

for k in k_values:
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers) #  κέντρα σημείων --> unit8

    black_background = np.zeros_like(football_image)

    for i in range(k):

        # τα pixel της εικόνας που θα εμφανιστού με την πράξη --> AND.
        # εάν pixel στην mask = 1 --> ίδιο στην εικόνα
        # εάν pixel στην mask = 0 --> μαύρο στην εικόνα 
        mask = np.uint8(labels == i)
        mask_resized = cv2.resize(mask, (football_image.shape[1], football_image.shape[0])) 
        mask_resized = mask_resized.reshape(football_image.shape[0], football_image.shape[1], -1)
        mask_resized = np.uint8(mask_resized)
        
        segment = cv2.bitwise_and(football_image, football_image, mask=mask_resized)
        cv2.imshow(f'Segment {i+1} with k={k}', segment)


cv2.imshow('football image', football_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
