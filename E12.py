import cv2
import numpy as np

# Φόρτωση της εικόνας
image = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Υπολογισμός διαστάσεων εικόνας
height, width = image.shape

# Δημιουργία μιας κενής εικόνας για την αποθήκευση των αποτελεσμάτων
compressed_image = np.zeros((height, width), dtype=np.float32)

# Εφαρμογή μετασχηματισμού DCT σε κάθε μη επικαλυπτόμενη περιοχή 32x32
for i in range(0, height, 32):
    for j in range(0, width, 32):
        # Περιοχή 32x32
        block = image[i:i+32, j:j+32]
        # Μετασχηματισμός DCT
        dct_block = cv2.dct(np.float32(block))
        # Αποθήκευση του μετασχηματισμένου block στην εικόνα συμπίεσης
        compressed_image[i:i+32, j:j+32] = dct_block

# Αποθήκευση της συμπιεσμένης εικόνας
cv2.imwrite('compressed_lenna.jpg', compressed_image)