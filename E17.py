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


cv2.imshow('coins image with threshold', coins_thresholded)

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