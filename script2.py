import onnxruntime as ort
import numpy as np
from PIL import Image
from copy import deepcopy
import time

# return unix timestamp on ms
def get_timestamp():
    return int(time.time() * 1000)

img = Image.open('/home/fibiaan/Downloads/2148946299.jpg').convert('RGB')
print(img.size)

orig_width, orig_height = img.size
resized_width, resized_height = img.size

if orig_width > orig_height:
    resized_width = 1024
    resized_height = int((1024 / orig_width) * orig_height)
else:
    resized_height = 1024
    resized_width = int((1024 / orig_height) * orig_width)

img = img.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
print(img.size)

input_tensor = np.array(img)

mean = np.array([123.675, 116.28, 103.53])
std = np.array([[58.395, 57.12, 57.375]])
input_tensor = (input_tensor - mean) / std

input_tensor = input_tensor.transpose(2,0,1)[None,:,:,:].astype(np.float32)

print(input_tensor.shape)


# Make image square 1024x1024 by padding short side by zeros
if resized_height < resized_width:
    input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,1024-resized_height),(0,0)))
else:
    input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,0),(0,1024-resized_width)))

print(input_tensor.shape)

encoder = ort.InferenceSession(
    '/home/fibiaan/Downloads/vit_b_encoder.onnx'
    )

start = get_timestamp()
output = encoder.run(None, {'images': input_tensor})
end = get_timestamp()
print(output)
embedding = output[0]

np.save("/home/fibiaan/Downloads/embeddings.npy",embedding)
print(f"Time taken: {end-start}ms")

input_point = np.array([[750, 456]])
input_label = np.array([1])

onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])])[None, :].astype(np.float32)

coords = deepcopy(onnx_coord).astype(float)
coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
coords[..., 1] = coords[..., 1] * (resized_height / orig_height)

onnx_coord = coords.astype("float32")

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

decoder = ort.InferenceSession("/home/fibiaan/Downloads/sam_vit_b_decoder.onnx")
masks,_,_ = decoder.run(None,{
    "image_embeddings": embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array([orig_height, orig_width], dtype=np.float32)
})

print(masks)

for index,mask in enumerate(masks[0]):
    mask = (mask > 0).astype('uint8')*255
    img_mask = Image.fromarray(mask,"L")
    img_mask.save(f'output-{index}.png', format='PNG', optimize=True, antialias=True)

