import cv2
import numpy as np
import matplotlib.pyplot as plt


# hallway image , γκρι 
hallway = cv2.imread('hallway.png')
hallway_image = cv2.imread('hallway.png', cv2.IMREAD_GRAYSCALE)

x_sobel = cv2.Sobel(hallway_image, cv2.CV_64F, 1, 0, ksize=3) # sobel στον x
absolute_x_sobel = np.abs(x_sobel) # απόλυτη τιμή συνιστώσας

y_sobel = cv2.Sobel(hallway_image, cv2.CV_64F, 0, 1, ksize=3) # sobel στον y
absolute_y_sobel = np.abs(y_sobel)

image_gradient = np.sqrt(x_sobel**2 + x_sobel**2) # gradient



                    # ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ ΦΩΤΕΙΝΟΤΗΤΑΣ --> GAMMA CORRECTION
def gamma_correction(image, gamma):
    return np.power(image / np.max(image), gamma)

gamma = 0.3
#absolute_x_sobel_gamma = gamma_correction(absolute_x_sobel, gamma)
#absolute_y_sobel_gamma = gamma_correction(absolute_y_sobel, gamma)
image_gradient_gamma = gamma_correction(image_gradient, gamma)



                   # ΟΛΙΚΗ ΚΑΤΩΦΛΙΩΣΗ ΣΤΗΝ ΕΙΚΟΝΑ ΠΛΑΤΟΥΣ ΤΟΥ GRADIENT
threshold = 100  # τιμή κατωφλίου
_, thresholded_image = cv2.threshold(image_gradient_gamma, threshold / 255, 1, cv2.THRESH_BINARY)

thresholded_image = thresholded_image.astype(np.uint8)


                   # ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΣ HOUGH
Hough_lines = cv2.HoughLinesP(
    thresholded_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10
)

if Hough_lines is not None:
    for line in Hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hallway, (x1, y1), (x2, y2), (255, 0, 255), 2)
else:
    print("No lines.")



fig, axs = plt.subplots(1, 3, figsize=(24, 6))

axs[0].imshow(absolute_x_sobel, cmap='gray')
axs[0].set_title('absolute sobel x')
axs[0].axis('off')

axs[1].imshow(absolute_y_sobel, cmap='gray')
axs[1].set_title('absolute sobel y')
axs[1].axis('off')

#axs[2].imshow(image_gradient, cmap='gray')
#axs[2].set_title('Gradient (NO gamma correction)')
axs[2].imshow(image_gradient_gamma, cmap='gray')
axs[2].set_title('Gradient (gamma correction)')
axs[2].axis('off')


plt.figure()  

plt.hist(image_gradient_gamma.ravel(), bins=256, range=[0, 1], color='orange')
plt.axvline(x=threshold, color='red', linestyle='--')
plt.title('Histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')


plt.figure(figsize=(6, 6))
plt.imshow(thresholded_image, cmap='gray')
plt.title('Gradient image with threshold')
plt.axis('off')


plt.figure(figsize=(8, 4))
plt.imshow(hallway, cmap='gray')
plt.title('Hough algorithm')
plt.axis('off')

plt.show()