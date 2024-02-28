# from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np

# sam = sam_model_registry['vit_b'](checkpoint='/home/fibiaan/Downloads/sam_vit_b.pth')

image = Image.open('/home/fibiaan/Downloads/dog.jpeg')
image_array = np.array(image)

# predictor = SamPredictor(sam)
# predictor.set_image(image_array)
# mask, _, _ = predictor.predict()

# print(mask)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry['vit_b'](checkpoint='/home/fibiaan/Downloads/sam_vit_b.pth')
sam.to(device='cuda')

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_array)

print(masks)