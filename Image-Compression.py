import cv2
import numpy as np

# lenna.jpg
lenna_image = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

height, width = lenna_image.shape
block_size = 32 # block διαστάσεων 32x32 

dct_blocks = [cv2.dct(np.float32(lenna_image[y:y+block_size, x:x+block_size]))  # DCT για κάθε block

for y in range(0, height, block_size) for x in range(0, width, block_size)  if lenna_image[y:y+block_size, x:x+block_size].shape == (block_size, block_size)]

# για κάθε block στο dct_blocks --> εκτύπωση συντελεστών
for i, block in enumerate(dct_blocks):
    print(f"Οι συντελεστές του block {i} είναι:")
    print(block)

# block σε κάθε διάσταση 
number_blocks_x = int(width / block_size)
number_blocks_y = int(height / block_size)
# όλα τα block σε μία εικόνα
rows_blocks = [np.hstack(dct_blocks[i*number_blocks_x  : (i+1)*number_blocks_x ]) for i in range(number_blocks_y)] # βάζει τα block στην ίδια γραμμή 
merged_blocks = np.vstack(rows_blocks) # βάζει τα block κατακόρυφα 
final_blocks_image = np.where(np.isnan(merged_blocks), 0, np.uint8(merged_blocks))# σε uint8 τύπο data, και τελική εικόνα με blocks


                            # ΜΕΘΟΔΟΣ ΖΩΝΗΣ 
p = 0.9  # συντελεστές που θα κρατήσω  (ή 0.1)

number_of_coef = int(block_size * block_size * p) # 32 * 32 * p --> συντελεστές που θα κρατήσω σε κάθε block
the_mask = np.zeros((block_size, block_size), dtype=np.float32) # array 32*32 μάσκα

indices = np.triu_indices(block_size)  # στο άνω τριγωνικό κρατάμε συντ. 
the_mask[indices[0][:number_of_coef ], indices[1][:number_of_coef ]] = 1 

blocks_with_coef = [block * the_mask for block in dct_blocks] # μάσκα σε κάθε block --> block με συντ. που κρατάω

blocks_in_rows = [np.hstack(blocks_with_coef[i*number_blocks_x: (i+1)*number_blocks_x]) for i in range(number_blocks_y)] # block x
merged_blocks1 = np.vstack(blocks_in_rows) # block y+x

idct_blocks = [cv2.idct(block) for block in blocks_with_coef] # idct

idct_blocks_in_rows = [np.hstack(idct_blocks[i*number_blocks_x: (i+1)*number_blocks_x]) for i in range(number_blocks_y)]
merged = np.vstack(idct_blocks_in_rows)
image_zonal_coding = np.uint8(np.clip(merged, 0, 255))


                        # ΜΕΘΟΔΟΣ ΚΑΤΩΦΛΙΟΥ
p1 = 1 # ποσοστό p της πληροφορίας που θα κρατήσω
# κατωφλι --> για τη μέθοδο της ζώνης
# εάν συντελεστές > τιμή κατωφλίου --> κρατάμε
threshold = np.percentile(np.abs(merged_blocks), 100 * (1 - p1))
number_of_coef1 = np.where(np.abs(merged_blocks) > threshold, merged_blocks, 0)

print(f"Οι συντελεστές από threshold coding για p = {p1}:")

print(number_of_coef1)

# idct
idct_blocks_thres = [cv2.idct(block) for block in number_of_coef1]

idct_rows_thres = [np.hstack(row_blocks) for row_blocks in idct_blocks_thres] # block σε γραμμές στήλες
merged1 = np.vstack(idct_rows_thres)
image_threshold_coding = np.uint8(np.clip(merged1, 0, 255))



                    # ΥΠΟΛΟΓΙΣΜΟΣ MSE

values_of_p = np.linspace(0.05, 0.50, num=10)













# εκτυπώσεις 
cv2.imshow('lenna image', lenna_image)
cv2.imshow('2D-DCT', final_blocks_image)
cv2.imshow('Zonal Method', image_zonal_coding)
cv2.imshow('Threshold Method', image_threshold_coding)

cv2.waitKey(0)
cv2.destroyAllWindows()
