from imports import *
from huggingface_hub import hf_hub_download
from model import ResUNetDeeper

def load_from_hf(model_path):
    hf_model_path = hf_hub_download(repo_id = "ramshakhan/segmentation-model", filename = model_path, )
    print(hf_model_path)
    model = ResUNetDeeper()
    checkpoint = torch.load(hf_model_path, map_location=torch.device('cpu')) # for cpu device
    model.load_state_dict(checkpoint['model_state_dict'])
    return model # pass model for onnx conversion then quantization later.