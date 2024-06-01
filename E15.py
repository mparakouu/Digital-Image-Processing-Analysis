import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import wiener

# new_york_image
newYork_image = cv2.imread('new_york.png')
cv2.imshow('New York image', newYork_image)

# BGR --> RGB 
newYork_image_rgb = cv2.cvtColor(newYork_image, cv2.COLOR_BGR2RGB)
standard_deviation = 2.0  # τυπική απόκλιση από [1.5, 2.0]
# 2D πυρήνα εξομάλυνσης Gauss filter
final_blurred_image = cv2.GaussianBlur(newYork_image_rgb, (0, 0), standard_deviation)



                        # ΠΡΟΣΘΗΚΗ ΛΕΥΚΟΥ ΘΟΡΥΒΟΥ GAUSSIAN
power = np.mean(final_blurred_image ** 2)  # ενέργεια σήματος
variance_noise = power / (10 ** (7 / 10))  # διασπορά θορύβου
white_Gaussian_noise = np.random.normal(0, np.sqrt(variance_noise), final_blurred_image.shape).astype(np.uint8)
noisy_image = cv2.add(final_blurred_image, white_Gaussian_noise)  # white gaussian noise add --> final_blurred_image
print("Shape of noisy_image:", noisy_image.shape)


# οι τιμές των pixel --> εύρος [0, 255]
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)



                    # ΚΡΟΥΣΤΙΚΗ ΑΠΟΚΡΙΣΗ & ΑΠΟΚΡΙΣΗ ΣΥΧΝΟΤΗΤΑΣ ΤΟΥ ΦΙΛΤΡΟΥ ΥΠΟΒΑΘΜΙΣΗΣ 
kernel_size = 9  # πυρήνα φίλτρου υποβάθμισης (9 x 9)
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
filtered_image = cv2.filter2D(final_blurred_image, -1, kernel)

print("Shape of kernel:", np.ones([kernel_size, kernel_size]).shape)


fourier_kernel = np.fft.fft2(kernel, s=(256, 256)) # κρουστική απόκριση --> # fourier στον πυρήνα
fourier_shift_kernel = np.fft.fftshift(fourier_kernel)  # μετατόπιση του μετασχ. πυρήνα στο κέντρο 
Frequency_response_kernel = 20 * np.log(np.abs(fourier_shift_kernel) + 1e-10) # απόκριση συχνότητας 


                    # WIENER DECONVOLUTION
gray_noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)
estimated_image = wiener(gray_noisy_image, mysize=[kernel_size, kernel_size], noise=variance_noise)



cv2.imshow('New York image', newYork_image_rgb)
cv2.imshow('Filter the new york image using: 2D Gaussian smoothing', final_blurred_image)
cv2.imshow('Add white Gaussian noise to blurred image', noisy_image)
cv2.imshow('Wiener Deconvolution Result', estimated_image)


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



cv2.waitKey(0)
cv2.destroyAllWindows()
