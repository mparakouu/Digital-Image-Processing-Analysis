import cv2
import numpy as np

# leaf image
leaf_image = cv2.imread('leaf.jpg')

rows, columns, _ = leaf_image.shape
rotation_matrix = cv2.getRotationMatrix2D((columns/2, rows/2), 60, 1) # περιστροφή κάτα 60 μοίρες
rotated_image = cv2.warpAffine(leaf_image, rotation_matrix, (columns, rows))

# Ολίσθηση εικόνας κατά x = 100 και y = 60
x, y = 100, 60

# Πόσο θα μετακινηθεί στον άξονα x, y 
# [1, 0, x] --> μετακινεί στον x
# [0, 1, y] --> μετακινεί στον y
x_y_rotation_matrix = np.float32([[1, 0, x], [0, 1, y]]) 

image_rotation_matrix = cv2.warpAffine(rotated_image, x_y_rotation_matrix, (columns, rows))

grayscale_image = cv2.cvtColor(image_rotation_matrix, cv2.COLOR_BGR2GRAY) # grayscale 
_, binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY) # τ = 220 , binary


thresholded_image = cv2.bitwise_not(binary_image) # αντιστροφή της δυαδική 
cv2.imshow('Binary_image reverse', thresholded_image)  


                        # ΑΛΓΟΡΙΘΜΟΣ MOORE BOUNDARY TRACING 
# moore boundary tracing algorithm
def moore_boundary_algorithm(image):
    boundaries, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = boundaries[0] if boundaries else None
    return boundary

# εξάγουμε το περίγραμμα του φύλλου από τη δυαδική εικόνα
leaf_with_pink_boundary = moore_boundary_algorithm(thresholded_image)

color = (255, 0, 255) # χρώμα για το περίγραμμα --> ροζ 
# σχειδάζουμε το ροζ περίγραμμα πάνω στην leaf_image
cv2.drawContours(leaf_image, [leaf_with_pink_boundary], -1, color, thickness=2)
cv2.imshow('Moore boundary tracing algorithm with pink outline', leaf_image) # εμφάνιση leaf_image με ροζ περίγραμμα





cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Grayscale Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', leaf_image)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
