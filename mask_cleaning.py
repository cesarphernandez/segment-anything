import cv2
import numpy as np
import time

def ms_datetime():
    return int(time.time() * 1000)


start = ms_datetime()
# Load your mask image
mask = cv2.imread('/home/fibiaan/Downloads/image.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Thresholding
_, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# Step 2: Morphological Operations
kernel = np.ones((5, 5), np.uint8)
# You can adjust the kernel size based on the size of the features you want to remove
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 3: Connected Component Analysis (Optional)
# If there are small connected components to remove, you can use connected component analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
for i in range(1, num_labels):
    # Remove small connected components by area thresholding
    if stats[i, cv2.CC_STAT_AREA] < 400:
        cleaned_mask[labels == i] = 0

# Step 4: Region Growing (Optional)
# If you need to expand the concentrated area, you can use region growing algorithms

end = ms_datetime()

print(f"Time taken: {end - start}ms")

# Save the cleaned mask
cv2.imwrite('cleaned_mask.png', cleaned_mask)
