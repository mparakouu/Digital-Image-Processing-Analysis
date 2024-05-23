import cv2
import numpy as np

# διαβάζουμε την εικόνα
leaf_image = cv2.imread('leaf.jpg')

# μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)

# κατωφλιώνουμε με τιμή κατωφλίου τ = 220
_, thresholded_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# εκτυπώνουμε τις εικόνες
# αρχική εικόνα leaf_image
cv2.imshow('leaf_image', leaf_image)
# grayscale image show 
cv2.imshow('Grayscale_image', grayscale_image)
# δυαδική image show 
cv2.imshow('Thresholded_image', thresholded_image)  

# αντιστροφή της δυαδική 
thresholded_image = cv2.bitwise_not(thresholded_image)

# moore boundary tracing algorithm
def moore_boundary_algorithm(image):
    boundaries, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = boundaries[0] if boundaries else None
    return boundary


# εξάγουμε το περίγραμμα του φύλλου από τη δυαδική εικόνα
leaf_with_pink_boundary = moore_boundary_algorithm(thresholded_image)

# χρώμα για το περίγραμμα --> ροζ
color = (255, 0, 255)  

# σχειδάζουμε το ροζ περίγραμμα πάνω στην leaf_image
cv2.drawContours(leaf_image, [leaf_with_pink_boundary], -1, color, thickness=2)

# εμφάνιση leaf_image με ροζ περίγραμμα
cv2.imshow('Moore boundary tracing algorithm with pink outline', leaf_image)

# υπολογισμός --> περιγραφείς Fourier του ροζ περιγράμματος
# ροζ περίγραμμα --> σύνθετους αριθμούς
leaf_pink_boundary_to_complex = np.array(leaf_with_pink_boundary[:, 0, 0] + 1j * leaf_with_pink_boundary[:, 0, 1], dtype=np.complex128)

# υπολογισμός : μετασχηματισμος Fourier του πίνακα με τους αριθμούς των ροζ περιγράματος
fourier_description = np.fft.fft(leaf_pink_boundary_to_complex)

# Εκτυπώνουμε τους περιγραφείς Fourier / πίνακας συντελεστών DFT
print("Περιγραφείς Fourier του ροζ περιγράμματος:")
print(fourier_description)


# Ανακατασκευή του ροζ περιγράμματος από τους περιγραφείς fourier με IFFT

# σημαντικότεροι συντελεστές
percentages = [100, 50, 10, 1]

# IFFT για κάθε ποσοστό 
for percentage in percentages:

    # πόσους συντελεστές χρησιμοποιούμε κάθε φορά
    number_of_coefficients = int(len(fourier_description) * percentage / 100)
    
    # IFFT στους συντελεστές fourier
    IFFT_description_fourier = np.fft.ifft(fourier_description[:number_of_coefficients])
    
    # μεττροπή των σύνθετων αριθμών σε του IFFT --> σημεία του ροζ περιγράμματος 
    reconstructed_image_outline = np.array([np.real(IFFT_description_fourier), np.imag(IFFT_description_fourier)]).T.reshape((-1, 1, 2)).astype(np.int32)
    
    # στην αρχική εικόα 
    reconstructed_image = cv2.imread('leaf.jpg')

    # χρώμα για το περίγραμμα του ανακατασκευασμένου --> μωβ
    color1 = (150, 0, 150)    
    
    # σχεδίαση του περιγράμματος πάνω στην εικόνα 
    cv2.drawContours(reconstructed_image, [reconstructed_image_outline], -1, color1, thickness=2)

    # Εμφάνιση της ανακατασκευασμένης εικόνας
    cv2.imshow(f'Reconstructed image {percentage}% of most significant coefficients', reconstructed_image)



cv2.waitKey(0)
cv2.destroyAllWindows()
