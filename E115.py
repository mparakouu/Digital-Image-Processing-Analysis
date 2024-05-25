import cv2
import numpy as np

# διαβάζουμε την εικόνα
image = cv2.imread('leaf.jpg')

# περιστροφή εικόνας κατά 60 μοίρες
rows, columns, _ = image.shape
center_rotation = (columns/2, rows/2)  
angle_rotation = 60  
scale = 1  
rotation_matrix = cv2.getRotationMatrix2D(center_rotation, angle_rotation, scale)
rotated_image = cv2.warpAffine(image, rotation_matrix, (columns, rows))

# ολίσθηση εικόνας κατά -->  x = 100 και y = 60
x, y = 100, 60

# πόσο θα μετακινηθεί στον άξονα x , y 
# [1, 0, x] --> μετακινεί στον x
# [0, 1, y] --> μετακινεί στον y
x_y_rotation_matrix = np.float32([[1, 0, x], [0, 1, y]]) 

image_rotation_matrix = cv2.warpAffine(rotated_image, x_y_rotation_matrix, (columns, rows))

# μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(image_rotation_matrix, cv2.COLOR_BGR2GRAY)

# κατωφλιώνουμε με τιμή κατωφλίου τ = 220
_, binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)


# Εμφανίζουμε τις εικόνες
cv2.namedWindow('Grayscale image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Binary image', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()