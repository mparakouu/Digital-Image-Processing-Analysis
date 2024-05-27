import cv2
import matplotlib.pyplot as plt

# board_image
road_image1 = cv2.imread('dark_road_1.jpg')
road_image2 = cv2.imread('dark_road_2.jpg')
road_image3 = cv2.imread('dark_road_3.jpg')

gray_road_image1 = cv2.cvtColor(road_image1, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_road_image1', gray_road_image1)
hist1 = cv2.calcHist([gray_road_image1], [0], None, [256], [0, 256])

gray_road_image2 = cv2.cvtColor(road_image2, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_road_image2', gray_road_image2)
hist2 = cv2.calcHist([gray_road_image2], [0], None, [256], [0, 256])

gray_road_image3 = cv2.cvtColor(road_image3, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_road_image3', gray_road_image3)
hist3 = cv2.calcHist([gray_road_image3], [0], None, [256], [0, 256])


                    # ΟΛΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
equalizeHist1 = cv2.equalizeHist(gray_road_image1)
cv2.imshow('Dark road image 1 EqualizedHist', equalizeHist1) 

equalizeHist2 = cv2.equalizeHist(gray_road_image2)
cv2.imshow('Dark road image 2 EqualizedHist', equalizeHist2)

equalizeHist3 = cv2.equalizeHist(gray_road_image3)
cv2.imshow('Dark road image 3 EqualizedHist', equalizeHist3)

finalHist1 = cv2.calcHist([equalizeHist1], [0], None, [256], [0, 256])
finalHist2 = cv2.calcHist([equalizeHist2], [0], None, [256], [0, 256])
finalHist3 = cv2.calcHist([equalizeHist3], [0], None, [256], [0, 256])


                            # ΤΟΠΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equalize_Hist1 = clahe.apply(gray_road_image1)
equalize_Hist2 = clahe.apply(gray_road_image2)
equalize_Hist3 = clahe.apply(gray_road_image3)




                        #ΕΚΤΥΠΩΣΗ HISTOGRAM
plt.figure(figsize=(8, 4))


plt.subplot(1, 3, 1)
plt.plot(hist1, color='red')
plt.title('Dark road image 1 histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(hist2, color='pink')
plt.title('Dark road image 2 histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.plot(hist3, color='purple')
plt.title('Dark road image 3 histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')


                        # ΕΚΤΥΠΩΣΗ ΤΩΝ ΟΛΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.plot(finalHist1, color='red')
plt.title('Dark road image 1 EqualizedHist')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(finalHist2, color='pink')
plt.title('Dark road image 2 EqualizedHist')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.plot(finalHist3, color='purple')
plt.title('Dark road image 3 EqualizedHist')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')


                        #ΕΚΤΥΠΩΣΗ ΤΩΝ # ΤΟΠΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
plt.figure(figsize=(8, 4))


plt.subplot(2, 3, 1)
plt.imshow(equalize_Hist1, cmap='gray')
plt.title('Road Image 1')
plt.subplot(2, 3, 4)
plt.hist(equalize_Hist1.flatten(), bins=256, range=[0,256], color='red')
plt.title('Histogram for equal. Road Image 1')


plt.subplot(2, 3, 2)
plt.imshow(equalize_Hist2, cmap='gray')
plt.title('Road Image 2')
plt.subplot(2, 3, 5)
plt.hist(equalize_Hist2.flatten(), bins=256, range=[0,256], color='pink')
plt.title('Histogram for equal. Road Image 2')


plt.subplot(2, 3, 3)
plt.imshow(equalize_Hist3, cmap='gray')
plt.title('Road Image 3')
plt.subplot(2, 3, 6)
plt.hist(equalize_Hist3.flatten(), bins=256, range=[0,256], color='purple')
plt.title('Histogram for equal. Road Image 3')

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()