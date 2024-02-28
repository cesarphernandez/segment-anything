import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import gc
gc.collect()
torch.cuda.empty_cache()

def show_output(result_dict,axes=None):
     if axes:
        ax = axes
     else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
     sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
     # Plot for each segment area
     for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
MODEL_TYPE = 'vit_b'
model_path = '/home/fibiaan/Downloads/sam_vit_b.pth'

sam = sam_model_registry[MODEL_TYPE](checkpoint=model_path).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_NAME = '/home/fibiaan/Downloads/dog.jpeg'

image_bgr = cv2.imread(IMAGE_NAME)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)
# _,axes = plt.subplots(1,2, figsize=(16,16))
# axes[0].imshow(image_rgb)
# show_output(sam_result, axes[1])
# plt.show()




