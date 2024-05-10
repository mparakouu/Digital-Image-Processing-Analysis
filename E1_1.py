import cv2
import numpy as np

# διαβάζουμε την εικόνα
leaf_image = cv2.imread('leaf.jpg')

# μετατρέπουμε την εικόνα σε grayscale
grayscale_image = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)

# κατωφλιώνουμε με τιμή κατωφλίου τ = 220
_, thresholded_image = cv2.threshold(grayscale_image, 220, 255, cv2.THRESH_BINARY)

# αρχική εικόνα leaf_image
cv2.imshow('leaf_image', leaf_image)

# grayscale image show 
cv2.imshow('Grayscale_image', grayscale_image)

# δυαδική image show 
cv2.imshow('Thresholded_image', thresholded_image)  

# σε δυαδική 
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

# υπολογισμός περιγραφέων Fourier του περιγράμματος
fourier_description = cv2.dft(leaf_with_pink_boundary.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

# σχειδάζουμε το ροζ περίγραμμα πάνω στην leaf_image
cv2.drawContours(leaf_image, [leaf_with_pink_boundary], -1, color, thickness=2)

# εμφάνιση leaf_image με ροζ περίγραμμα
cv2.imshow('Moore boundary tracing algorithm with pink boundary', leaf_image)

# περιγραφείς Fourier του boundary
print("Περιγραφείς Fourier του περιγράμματος:")
print(fourier_description)

# συνάρτηση ανακατασκευής του περιγράμματος του φύλλου από τις περιγραφείς fourier 
def reconstruct_boundary(fourier_descript, percent):

    number_of_coefficients = fourier_descript.shape[0]  # Πλήθος συντελεστών Fourier

    num_to_keep = int(percent / 100 * number_of_coefficients)  # Πλήθος συντελεστών που θα διατηρηθούν

    sorted_magnitudes = np.argsort(np.abs(fourier_descript[:, 0, 0]))[::-1]  # Ταξινόμηση των συντελεστών με φθίνουσα σειρά απόλυτης τιμής

    to_keep_indices = sorted_magnitudes[:num_to_keep]  # Επιλογή των πρώτων συντελεστών με τις μεγαλύτερες απόλυτες τιμές

    reconstruction = np.zeros_like(fourier_descript)  # Δημιουργία μηδενικού πίνακα με τις ίδιες διαστάσεις με τους συντελεστές Fourier

    reconstruction[to_keep_indices] = fourier_descript[to_keep_indices]  # Ανακατασκευή περιγράμματος χρησιμοποιώντας μόνο τους επιλεγμένους συντελεστές

    reconstructed_boundary = cv2.idft(reconstruction, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # Αντίστροφος μετασχηματισμός Fourier

    return reconstructed_boundary.astype(np.int32)  # Επιστροφή ανακατασκευασμένου περιγράμματος ως ακέραιους αριθμούς

# τα ποσοστά των συντελεστών που θα χρησιμοποιηθού --> 100% , 50% , 10% , 1%
percentages = [100, 50, 10, 1]

# βρίσκω για κάθε ποστοστό 
for percent in percentages:
    
    reconstructed_boundary = reconstruct_boundary(fourier_description, percent)
    reconstructed_boundary = np.expand_dims(reconstructed_boundary, axis=1)  # Προσθήκη διαστάσεων
    reconstructed_image = np.zeros_like(leaf_image)  # Δημιουργία μιας μαύρης εικόνας με τις ίδιες διαστάσεις με την αρχική
    cv2.drawContours(reconstructed_image, [reconstructed_boundary], -1, color, thickness=2)  # Σχεδίαση του περιγράμματος στην μαύρη εικόνα
    cv2.imshow(f'Reconstructed contour {percent}%', reconstructed_image)  # Εμφάνιση του ανακατασκευασμένου περιγράμματος

cv2.waitKey(0)
cv2.destroyAllWindows()
