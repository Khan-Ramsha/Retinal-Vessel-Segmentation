"""
Converting torch model to onnx format. Loading trained model from huggingface hub

"""
from imports import *

def convert_onnx(model):
    model.eval()
    model = model.cpu()
    os.makedirs("models", exist_ok=True)
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(model, dummy_input, "models/resunet.onnx", export_params = True, opset_version = 17, input_names = ['input'], output_names = ['output'], dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    ## MODEL accepts image size of 512*512

    onnx_model = onnx.load("models/resunet.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked successfully!")
    size_mb = os.path.getsize("models/resunet.onnx") / (1024**2)
    print(f"ONNX model size: {size_mb:.1f} MB")
    return onnx_model