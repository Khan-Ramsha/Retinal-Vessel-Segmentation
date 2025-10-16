
#-------Converting fp32 to fp16-----------

from imports import *
def quantize_model():
    os.makedirs("models", exist_ok=True)  # Create directory
    onnx_model = onnx.load("models/resunet.onnx")
    model_fp16 = float16.convert_float_to_float16(onnx_model)
    onnx.save(model_fp16, "models/quantized_model.onnx")
    print("Quantized model saved to models/quantized_model.onnx")
