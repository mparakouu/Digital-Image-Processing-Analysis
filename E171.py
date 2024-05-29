import cv2
import numpy as np
import matplotlib.pyplot as plt


# coins image 
coins_image = cv2.imread('coins.png', cv2.IMREAD_GRAYSCALE)

threshold = 150
_, coins_thresholded = cv2.threshold(coins_image, threshold, 255, cv2.THRESH_BINARY)


                            # OTSU METHOD
histogram = cv2.calcHist([coins_image], [0], None, [256], [0, 256])
_, otsu_threshold = cv2.threshold(coins_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


                            # ΜΑΥΡΟ ΦΟΝΤΟ , ΑΠΟΤΕΛΕΣΜΑ ΔΥΑΔΙΚΗΣ ΚΑΤΩΦΛΙΩΣΗ ΩΣ ΜΑΣΚΑ
coins_black_background = cv2.bitwise_and(coins_image, coins_image, mask=otsu_threshold)
cv2.imshow('Black background with mask from otsu method:', coins_black_background)


coins_contours, _ = cv2.findContours(otsu_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # περίγραμμα κερμάτων
coins_white_background = np.ones((coins_image.shape[0], coins_image.shape[1], 3), dtype=np.uint8) * 255


                            # ΧΡΩΜΑΤΑ ΣΤΑ COINS
# χρώματα για κάθε coin --> πράσινο , μπλε, πορτοκαλί , λαδί , μπορντό , κίτρινο , ροζ , μωβ , κόκκινο , γκρι
coins_colors = [ (0, 255, 0),  (0, 0, 255),  (0, 165, 255), (128, 128, 0), (255, 0, 255), (0, 255, 255), (203, 192, 255), (128, 0, 128), (255, 0, 0), (128, 128, 128)  ]

for i, contour in enumerate(coins_contours):
    color = coins_colors[i % len(coins_colors)] # κάθε κέρμα έχει διαφορετικό χρώματα από λίστα colors
    cv2.drawContours(coins_white_background, [contour], -1, color, -1)



cv2.imshow('coins image with threshold', coins_thresholded)
cv2.imshow('coins with different colors & white background', coins_white_background)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Histogram')
plt.plot(histogram, color='purple')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.title('Otsu method thresholding')
plt.imshow(coins_image, cmap='gray')
plt.contour(otsu_threshold, colors='purple', levels=[1])

plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()