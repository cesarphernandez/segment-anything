import torch
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import gc
gc.collect()
torch.cuda.empty_cache()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = 'vit_h'
model_path = '/home/fibiaan/Downloads/sam_vit_h_4b8939.pth'

sam = sam_model_registry[MODEL_TYPE](checkpoint=model_path).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_NAME = '/home/fibiaan/Downloads/kraft.png'

image_bgr = cv2.imread(IMAGE_NAME)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)


