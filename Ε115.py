import cv2
import numpy as np

# Διάβασμα της εικόνας
leaf_image = cv2.imread('leaf.jpg')

# Ορίζουμε τη γωνία περιστροφής και τη μετατόπιση
angle = 60
translation = (100, 60)

# Υπολογισμός του πίνακα περιστροφής και μετατόπισης
rows, cols, _ = leaf_image.shape
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotation_matrix[:, 2] += translation

# Εφαρμογή περιστροφής και μετατόπισης στην εικόνα
transformed_image = cv2.warpAffine(leaf_image, rotation_matrix, (cols, rows))

# Μετατροπή της εικόνας σε grayscale
grayscale_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

# Κατωφλίωση της εικόνας
_, thresholded_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# Αντιστροφή της δυαδικής εικόνας
thresholded_image = cv2.bitwise_not(thresholded_image)

# Αλγόριθμος Moore Boundary
def moore_boundary_algorithm(image):
    boundaries, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = boundaries[0] if boundaries else None
    return boundary

# Εξαγωγή περιγράμματος
leaf_with_pink_boundary = moore_boundary_algorithm(thresholded_image)

# Σχεδίαση του περιγράμματος πάνω στην εικόνα
color = (255, 0, 255)
cv2.drawContours(transformed_image, [leaf_with_pink_boundary], -1, color, thickness=2)

# Εμφάνιση της εικόνας με το περίγραμμα
cv2.imshow('Moore Boundary with Pink Outline', transformed_image)

# Υπολογισμός περιγραφέων Fourier
leaf_pink_boundary_to_complex = np.array(leaf_with_pink_boundary[:, 0, 0] + 1j * leaf_with_pink_boundary[:, 0, 1], dtype=np.complex128)
fourier_description = np.fft.fft(leaf_pink_boundary_to_complex)

# Ανακατασκευή του περιγράμματος από περιγραφείς Fourier
percentages = [100, 50, 10, 1]

for percentage in percentages:
    number_of_coefficients = int(len(fourier_description) * percentage / 100)
    IFFT_description_fourier = np.fft.ifft(fourier_description[:number_of_coefficients])
    reconstructed_image_outline = np.array([np.real(IFFT_description_fourier), np.imag(IFFT_description_fourier)]).T.reshape((-1, 1, 2)).astype(np.int32)
    reconstructed_image = cv2.imread('leaf.jpg')
    color1 = (150, 0, 150)
    cv2.drawContours(reconstructed_image, [reconstructed_image_outline], -1, color1, thickness=2)
    cv2.imshow(f'Reconstructed image {percentage}% of most significant coefficients', reconstructed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
