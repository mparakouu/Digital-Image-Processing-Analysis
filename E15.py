import cv2
import numpy as np
from matplotlib import pyplot as plt

# new_york_image
newYork_image = cv2.imread('new_york.png')

# BGR --> RGB 
newYork_image_rgb = cv2.cvtColor(newYork_image, cv2.COLOR_BGR2RGB)
standard_deviation = 1.9 # τυπική απόκλιση από [1.5, 2.0]
# Φιλτράρισμα με 2D πυρήνα εξομάλυνσης Gauss
final_blurred_image = cv2.GaussianBlur(newYork_image_rgb, (0, 0), standard_deviation)


                    # ΠΡΟΣΘΗΚΗ ΛΕΥΚΟΥ ΘΟΡΥΒΟΥ GAUSSIAN
power = np.mean(final_blurred_image ** 2) #ενέργεια σήματος
variance_noise = power / (10 ** (7 / 10)) # διασπορά θορύβου
white_Gaussian_noise = np.random.normal(0, np.sqrt(variance_noise), final_blurred_image.shape).astype(np.uint8)
noisy_image = cv2.add(final_blurred_image, white_Gaussian_noise) # white gaussian noise add --> final_blurred_image

# Εξασφάλιση ότι οι τιμές των pixel παραμένουν στο επιτρεπτό εύρος [0, 255]
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)


                    # ΚΡΟΥΣΤΙΚΗ ΑΠΟΚΡΙΣΗ ΤΟΥΦ ΦΙΛΤΡΟΥ ΥΠΟΒΑΘΜΙΣΗΣ 
# Δημιουργία πυρήνα φίλτρου υποβάθμισης
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Υπολογισμός κρουστικής απόκρισης φίλτρου
f_kernel = np.fft.fft2(kernel)
fshift = np.fft.fftshift(f_kernel)
magnitude_spectrum_kernel = 20 * np.log(np.abs(fshift))

# Εμφάνιση αρχικής εικόνας, εικόνας με θόρυβο και κρουστικής απόκρισης φίλτρου
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(newYork_image_rgb)
plt.title('Original Image'), plt.axis('off')

plt.subplot(132), plt.imshow(noisy_image)
plt.title('Image with Noise'), plt.axis('off')

plt.subplot(133), plt.imshow(magnitude_spectrum_kernel, cmap='gray')
plt.title('Frequency Response of Filtering'), plt.axis('off')

plt.show()



cv2.imshow('New York image', newYork_image_rgb)
cv2.imshow('Filter the new york image using: 2D Gaussian smoothing', final_blurred_image)
cv2.imshow('Add white Gaussian noise to blurred image', noisy_image)




cv2.waitKey(0)
cv2.destroyAllWindows()