from load_model_from_hf import load_from_hf
from convert_to_onnx import convert_onnx
from quantize import quantize_model
from inference import test_quantized_onnx, data_loader, criterion, device
import os

model_path = "resunet_model.pth"

model = load_from_hf(model_path)
onnx_model = convert_onnx(model)
quantize_model()
val_loss, val_iou, val_dice, val_acc, avg_time, outputs = test_quantized_onnx("models/quantized_model.onnx",data_loader, criterion, device)

print(f"\nResults:")
print(f"Loss: {val_loss:.4f}")
print(f"IoU: {val_iou:.4f}")
print(f"Dice: {val_dice:.4f}")
print(f"Accuracy: {val_acc:.4f}")
print(f"Avg inference time: {avg_time:.4f}s")

# Check model sizes
fp32_size = os.path.getsize("models/resunet.onnx") / (1024**2)
fp16_size = os.path.getsize("models/quantized_model.onnx") / (1024**2)

print(f"\nModel sizes:")
print(f"FP32: {fp32_size:.2f} MB")
print(f"FP16: {fp16_size:.2f} MB")
print(f"Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%")