import cv2
import numpy as np

# εικόνα 
board_image = cv2.imread('board.png')

signal_board_image = np.mean(board_image ** 2) # σήμα εικόνας

variance_noise = signal_board_image / (10 ** (15 / 10)) # διασπορά θορύβου --> σήμα / θόρυβο = 15db

white_Gaussian_noise = np.random.normal(0, np.sqrt(variance_noise), board_image.shape).astype(np.uint8) # λευκός Gaussian θόρυβος

noise_board_image = cv2.add(board_image, white_Gaussian_noise) # θόρυβο --> εικόνα

# φίλτρο κινουμένου μέσου (moving average filter) & φίλτρο μεσαίου (median filter)
kernel_size = 3  # kernel μέγεθος
moving_average_filter = cv2.blur(noise_board_image, (kernel_size, kernel_size))
median_filter = cv2.medianBlur(noise_board_image, kernel_size)

cv2.imshow('Board Image', board_image)
cv2.imshow('White Gaussian Noise Image', noise_board_image)
cv2.imshow('moving average filter to remove noise', moving_average_filter)
cv2.imshow('median filter to remove noise', median_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()







