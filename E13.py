import cv2
import numpy as np

# board_image
board_image = cv2.imread('board.png')


                            # ΠΡΟΣΘΗΚΗ ΛΕΥΚΟΥ GAUSSIAN ΘΟΡΥΒΟΥ
signal_board_image = np.mean(board_image ** 2) # σήμα εικόνας

variance_noise = signal_board_image / (10 ** (15 / 10)) # διασπορά θορύβου --> σήμα / θόρυβο = 15db

white_Gaussian_noise = np.random.normal(0, np.sqrt(variance_noise), board_image.shape).astype(np.uint8) # λευκός Gaussian θόρυβος

noise_board_image = cv2.add(board_image, white_Gaussian_noise) # θόρυβο --> εικόνα

# φίλτρο κινουμένου μέσου (moving average filter) & φίλτρο μεσαίου (median filter)
filter_size = 3 # διαστάσεις  # 3x3 ή 9x9
filter_size1 = 9 
#filter_size2 = 32

moving_average_filter = cv2.blur(noise_board_image, (filter_size1, filter_size1)) 
median_filter = cv2.medianBlur(noise_board_image, filter_size1)


                            # ΕΙΣΑΓΩΓΗ ΚΡΟΥΣΤΙΚΟΥ ΘΟΡΥΒΟΥ
# κρουστικός θόρυβος με πιθανότητα : 30%
kernel_noise = np.zeros(board_image.shape[:2], dtype=np.uint8)
cv2.randu(kernel_noise, 0, 100)  # με random εισαγωγή του κρουστικού , RGB
kernel_noise[kernel_noise < 30] = 255 # 30/100 ασπρα
kernel_noise[kernel_noise < 255] = 0 # υπόλοιπα μαύρα

# κρουστικός θόρυβος --> board_image
noise_mask = cv2.merge([kernel_noise] * 3)  # Δημιουργία μάσκας θορύβου
noisy_board_image1 = cv2.add(board_image, noise_mask)

# φίλτρο κινουμένου μέσου (moving average filter) & φίλτρο μεσαίου (median filter)
moving_average_filter1 = cv2.blur(noisy_board_image1, (filter_size, filter_size))
median_filter1 = cv2.medianBlur(noisy_board_image1, filter_size)


cv2.imshow('Board Image', board_image)
cv2.imshow('White Gaussian Noise Image', noise_board_image)
cv2.imshow('moving average filter to remove white gaussian noise', moving_average_filter)
cv2.imshow('median filter to remove white gaussian noise', median_filter)

cv2.imshow('moving average filter to remove impulse noise', moving_average_filter1)
cv2.imshow('Median Filter to remove impulse noise', median_filter1)

cv2.waitKey(0)
cv2.destroyAllWindows()







