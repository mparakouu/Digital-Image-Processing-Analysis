import cv2
import numpy as np

# lenna.jpg
lenna_image = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

height, width = lenna_image.shape
block_size = 32 # block διαστάσεων 32x32 

# DCT για κάθε block
dct_blocks = [cv2.dct(np.float32(lenna_image[y:y+block_size, x:x+block_size])) 

for y in range(0, height, block_size) for x in range(0, width, block_size) 
if lenna_image[y:y+block_size, x:x+block_size].shape == (block_size, block_size)]

# για κάθε block στο dct_blocks --> εκτύπωση συντελεστών
for i, block in enumerate(dct_blocks):
    print(f"Οι συντελεστές του block {i} είναι:")
    print(block)

# block σε κάθε διάσταση 
number_blocks_x = int(width / 32)
number_blocks_y = int(height / 32)
# όλα τα block σε μία εικόνα
rows_blocks = [np.hstack(dct_blocks[i*number_blocks_x  : (i+1)*number_blocks_x ]) for i in range(number_blocks_y)] # βάζει τα block στην ίδια γραμμή 
merged_blocks = np.vstack(rows_blocks) # βάζει τα block κατακόρυφα 
final_blocks_image = np.uint8(merged_blocks) # σε uint8 τύπο data, και τελική εικόνα με blocks

cv2.imshow('lenna image', lenna_image)
cv2.imshow('Blocks after 2D-DCT', final_blocks_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
