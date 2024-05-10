import cv2
import numpy as np

# Διαβάζουμε την εικόνα
leaf_image = cv2.imread('leaf.jpg')

# Μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)

# Κατωφλιώνουμε με τιμή κατωφλίου τ = 220
_, thresholded_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# αρχική εικόνα leaf_image
cv2.imshow('leaf_image', leaf_image)

# grayscale image show 
cv2.imshow('Grayscale_image', grayscale_image)

# δυαδική image show 
cv2.imshow('Thresholded_image', thresholded_image)

# Αντίστροφη δυαδική μετατροπή για να κάνουμε το φύλλο μαύρο και το φόντο λευκό
thresholded_image = cv2.bitwise_not(thresholded_image)

# moore boundary tracing algorithm
def moore_boundary_algorithm(image):
    boundaries, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = boundaries[0] if boundaries else None
    return boundary

# Εξάγουμε το περίγραμμα του φύλλου από τη δυαδική εικόνα
leaf_with_pink_boundary = moore_boundary_algorithm(thresholded_image)

# χρώμα για το περίγραμμα --> ροζ
color = (255, 0, 255)  

# Υπολογισμός περιγραφέων Fourier του περιγράμματος
fourier_descriptors = cv2.dft(leaf_with_pink_boundary.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

# Εκτύπωση των περιγραφέων Fourier
print("Περιγραφείς Fourier του περιγράμματος:")
print(fourier_descriptors)

# Σχειδάζουμε το ροζ περίγραμμα πάνω στην leaf_image
cv2.drawContours(leaf_image, [leaf_with_pink_boundary], -1, color, thickness=2)

# Εμφάνιση leaf_image με ροζ περίγραμμα
cv2.imshow('Moore boundary tracing algorithm with pink boundary', leaf_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
