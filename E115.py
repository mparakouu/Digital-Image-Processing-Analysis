import cv2
import numpy as np

# Διαβάζουμε την εικόνα
leaf_image = cv2.imread('leaf.jpg')

# Περιστρέφουμε την εικόνα κατά 60 μοίρες
rows, columns, _ = leaf_image.shape
center_rotation = (columns/2, rows/2)  
angle_rotation = 60  
scale = 1  
rotation_matrix = cv2.getRotationMatrix2D(center_rotation, angle_rotation, scale)
rotated_image = cv2.warpAffine(leaf_image, rotation_matrix, (columns, rows))

# Ολίσθηση εικόνας κατά x = 100 και y = 60
x, y = 100, 60

# Πόσο θα μετακινηθεί στον άξονα x, y 
# [1, 0, x] --> μετακινεί στον x
# [0, 1, y] --> μετακινεί στον y
x_y_rotation_matrix = np.float32([[1, 0, x], [0, 1, y]]) 

image_rotation_matrix = cv2.warpAffine(rotated_image, x_y_rotation_matrix, (columns, rows))

# Μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(image_rotation_matrix, cv2.COLOR_BGR2GRAY)

# Κατωφλίωση με τιμή κατωφλίου τ = 220
_, binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# Αντίστροφη των χρωμάτων στην εικόνα
inverted_image = cv2.bitwise_not(binary_image)


# Εμφάνιση εικόνων
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Grayscale Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', leaf_image)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
