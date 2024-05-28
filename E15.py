import cv2
import numpy as np
from matplotlib import pyplot as plt

# new_york_image
newYork_image = cv2.imread('new_york.png')

# BGR --> RGB 
newYork_image_rgb = cv2.cvtColor(newYork_image, cv2.COLOR_BGR2RGB)
standard_deviation = 1.9  # τυπική απόκλιση από [1.5, 2.0]
# 2D πυρήνα εξομάλυνσης Gauss filter
final_blurred_image = cv2.GaussianBlur(newYork_image_rgb, (0, 0), standard_deviation)



                        # ΠΡΟΣΘΗΚΗ ΛΕΥΚΟΥ ΘΟΡΥΒΟΥ GAUSSIAN
power = np.mean(final_blurred_image ** 2)  # ενέργεια σήματος
variance_noise = power / (10 ** (7 / 10))  # διασπορά θορύβου
white_Gaussian_noise = np.random.normal(0, np.sqrt(variance_noise), final_blurred_image.shape).astype(np.uint8)
noisy_image = cv2.add(final_blurred_image, white_Gaussian_noise)  # white gaussian noise add --> final_blurred_image

# οι τιμές των pixel --> εύρος [0, 255]
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)



                    # ΚΡΟΥΣΤΙΚΗ ΑΠΟΚΡΙΣΗ & ΑΠΟΚΡΙΣΗ ΣΥΧΝΟΤΗΤΑΣ ΤΟΥ ΦΙΛΤΡΟΥ ΥΠΟΒΑΘΜΙΣΗΣ 
kernel_size = 9  # πυρήνα φίλτρου υποβάθμισης (9 x 9)
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)


fourier_kernel = np.fft.fft2(kernel, s=(256, 256)) # κρουστική απόκριση --> # fourier στον πυρήνα
fourier_shift_kernel = np.fft.fftshift(fourier_kernel)  # μετατόπιση του μετασχ. πυρήνα στο κέντρο 
Frequency_response_kernel = 20 * np.log(np.abs(fourier_shift_kernel) + 1e-10) # απόκριση συχνότητας 


                    # WIENER DECONVOLUTION

SNR = 1 / variance_noise 


# Υπολογισμός του Wiener Filter
wiener_filter = np.ones_like(fourier_shift_kernel, dtype=np.complex128)

wiener_filter[:256, :256] = np.conj(fourier_shift_kernel) / (np.abs(fourier_shift_kernel) ** 2 + 1 / SNR)


# Εφαρμογή Wiener Deconvolution
restored_image = np.fft.ifft2(np.fft.fft2(noisy_image) * wiener_filter).real
restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)


cv2.imshow('New York image', newYork_image_rgb)
cv2.imshow('Filter the new york image using: 2D Gaussian smoothing', final_blurred_image)
cv2.imshow('Add white Gaussian noise to blurred image', noisy_image)

plt.figure(figsize=(8, 4))
plt.subplot(131), plt.imshow(newYork_image_rgb)
plt.title('Original Image'), plt.axis('off')

plt.subplot(132), plt.imshow(noisy_image)
plt.title('Image with Noise'), plt.axis('off')

plt.figure(figsize=(8, 4))

plt.subplot(122)
plt.imshow(Frequency_response_kernel, cmap='gray')
plt.title('Frequency response')
plt.axis('off')

plt.subplot(121)
plt.imshow(np.real(fourier_shift_kernel), cmap='gray', interpolation='none')
plt.title('fourier transf. kernel in the center')
plt.axis('off')

plt.show()

# Παρουσίαση της αποκατεστημένης εικόνας
cv2.imshow('Restored Image using Wiener Deconvolution', restored_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
