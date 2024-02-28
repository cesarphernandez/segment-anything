from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
import torch

sam = sam_model_registry['vit_b'](checkpoint='/home/fibiaan/Downloads/sam_vit_b.pth')

torch.onnx.export(
    f='/home/fibiaan/Downloads/vit_b_encoder.onnx',
    model=sam.image_encoder,
    args=torch.randn(1, 3, 1024, 1024),
    input_names=['images'],
    output_names=['embedding'],
    export_params=True
)


onnx_model = SamOnnxModel(sam, return_single_mask=False)
embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}

output_names = ["masks", "iou_predictions", "low_res_masks"]
torch.onnx.export(
    f="/home/fibiaan/Downloads/sam_vit_b_decoder.onnx",
    model=onnx_model,
    args=tuple(dummy_inputs.values()),
    input_names=list(dummy_inputs.keys()),
    output_names=output_names,
    dynamic_axes={
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"}
    },
    export_params=True,
    opset_version=17,
    do_constant_folding=True
)