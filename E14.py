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
finalHist1 = cv2.calcHist([equalizeHist1], [0], None, [256], [0, 256])

equalizeHist2 = cv2.equalizeHist(gray_road_image2)
cv2.imshow('Dark road image 2 EqualizedHist', equalizeHist2)
finalHist2 = cv2.calcHist([equalizeHist2], [0], None, [256], [0, 256])

equalizeHist3 = cv2.equalizeHist(gray_road_image3)
cv2.imshow('Dark road image 3 EqualizedHist', equalizeHist3)
finalHist3 = cv2.calcHist([equalizeHist3], [0], None, [256], [0, 256])



                            # ΤΟΠΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

equalize_Hist1 = clahe.apply(gray_road_image1)
cv2.imshow('Dark road image 1 CLAHE', equalize_Hist1) 
final_Hist1 = cv2.calcHist([equalize_Hist1], [0], None, [256], [0, 256])

equalize_Hist2 = clahe.apply(gray_road_image2)
cv2.imshow('Dark road image 2 CLAHE', equalize_Hist2) 
final_Hist2 = cv2.calcHist([equalize_Hist2], [0], None, [256], [0, 256])

equalize_Hist3 = clahe.apply(gray_road_image3)
cv2.imshow('Dark road image 3 CLAHE', equalize_Hist3) 
final_Hist3 = cv2.calcHist([equalize_Hist3], [0], None, [256], [0, 256])




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


                        #ΕΚΤΥΠΩΣΗ ΤΩΝ ΤΟΠΙΚΗ ΕΞΙΣΩΣΗ HISTOGRAM
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.plot(final_Hist1, color='red')
plt.title('Dark road image 1 CLAHE')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(final_Hist2, color='pink')
plt.title('Dark road image 2 CLAHE')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.plot(final_Hist3, color='purple')
plt.title('Dark road image 3 CLAHE')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()