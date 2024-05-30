import cv2
import numpy as np

# διαβάζουμε την εικόνα
leaf_image = cv2.imread('leaf.jpg')

# μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)

# πως προκύπτει η δυαδική εικόνα :
# κατωφλιώνουμε με τιμή κατωφλίου τ = 220
# εάν >  τ=220 --> άσπρο
# εάν <  τ=220 --> μαύρο
_, binary_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# εκτυπώνουμε τις εικόνες
# αρχική εικόνα leaf_image
cv2.imshow('leaf_image', leaf_image)
# grayscale image show 
cv2.imshow('Grayscale_image', grayscale_image)
# δυαδική image show 
cv2.imshow('Binary_image', binary_image)  


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


                        # ΠΕΡΙΓΡΑΦΕΙΣ FOURIER
# υπολογισμός --> περιγραφείς Fourier του ροζ περιγράμματος
# ροζ περίγραμμα --> σύνθετους αριθμούς
leaf_pink_boundary_to_complex = np.array(leaf_with_pink_boundary[:, 0, 0] + 1j * leaf_with_pink_boundary[:, 0, 1], dtype=np.complex128)

# υπολογισμός : μετασχηματισμος Fourier του πίνακα με τους αριθμούς των ροζ περιγράματος
fourier_description = np.fft.fft(leaf_pink_boundary_to_complex)


print("Περιγραφείς Fourier του ροζ περιγράμματος:") # περιγραφείς Fourier / πίνακας συντελεστών DFT
print(fourier_description)


                        # ΑΝΑΚΑΤΑΣΚΕΥΗ ΠΕΡΙΓΡΑΜΜΑΤΟΣ ΑΠΟ ΠΕΡΙΓΡΑΦΕΙΣ FOURIER (IFFT)
# σημαντικότεροι συντελεστές
# 100 : χρήση όλων των συντελεστών --> δεν χάνεται πληροφορία
# 50 --> χάνεται πληοροφορία
# 10 --> χάνεται πληοροφορία
# 1 --> χάνεται πληοροφορία
percentages = [100, 50, 10, 1]

# IFFT
for percentage in percentages:
    #συντελεστές που θα χρησιμοποιήσω
    number_of_coefficients = int(len(fourier_description) * percentage / 100) # συν. αριθμός συντ.  * ποσοστο / 100 
    
    coeff_fourier_description = np.zeros_like(fourier_description) # πίνακας με συντ.
    #διατηρούμε τους πιο σημαντικούς συντε --> αρχή κ τέλος
    # παίρνουμε τους τελευταίους συντ --> πιο σημαντικοί και τους βάζουμε στον νέο πίνακα
    coeff_fourier_description[:number_of_coefficients] = fourier_description[:number_of_coefficients]
    coeff_fourier_description[-number_of_coefficients:] = fourier_description[-number_of_coefficients:]
    
    # IFFT
    IFFT_description_fourier = np.fft.ifft(coeff_fourier_description)
    
    reconstructed_image_boundary = np.array([ # ανακατασκευασμένο περίγραμμα
        np.real(IFFT_description_fourier),
        np.imag(IFFT_description_fourier)
    ]).T.reshape((-1, 1, 2))
    
    reconstructed_image_boundary = np.clip(reconstructed_image_boundary, 0, [leaf_image.shape[1], leaf_image.shape[0]])
    reconstructed_image_boundary = reconstructed_image_boundary.astype(np.int32) # --> ακέραιους
    
    reconstructed_image = cv2.imread('leaf.jpg')

    color1 = (150, 0, 150) # χρώμα για το περίγραμμα του ανακατασκευασμένου --> μωβ
    cv2.drawContours(reconstructed_image, [reconstructed_image_boundary], -1, color1, thickness=2) # μωβ περίγραμμα
    cv2.imshow(f'Reconstructed image {percentage}% of most significant coefficients', reconstructed_image)



cv2.waitKey(0)
cv2.destroyAllWindows()
