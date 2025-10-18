from load_model_from_hf import load_from_hf
from convert_to_onnx import convert_onnx
from quantize import quantize_model
from inference import *
import os
import gradio as gr
import shutil
from PIL import Image

# val_loss, val_iou, val_dice, val_acc, avg_time, outputs = test_quantized_onnx("models/quantized_model.onnx",data_loader, criterion, device) #uncomment this for evaluation task. 
# infer_time = infer("models/quantized_model.onnx", infer_dataloader)

def gradio_inference(uploaded_images):
    if os.path.exists("user_imgs/input"):
        shutil.rmtree("user_imgs/input")
    os.makedirs("user_imgs/input", exist_ok=True)
    if os.path.exists("output_img"):
        shutil.rmtree("output_img")
    os.makedirs("output_img", exist_ok=True)

    for idx, img_path in enumerate(uploaded_images):
        img = Image.open(img_path.name)
        path = os.path.join("user_imgs/input", f"img_{idx}.png")
        img.save(path) 
    
    model_path = "resunet_model.pth"
    model = load_from_hf(model_path)
    onnx_model = convert_onnx(model)
    quantize_model()
    infer_dataset = InferDataset("user_imgs/input", transform=infer_transform)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

    infer_time = infer("models/quantized_model.onnx", infer_dataloader)
    
    output_paths = sorted([os.path.join("output_img", f) for f in os.listdir("output_img")])
    print(output_paths)
    return output_paths  

demo = gr.Interface(
    fn=gradio_inference,
    inputs=gr.File(
            file_count="multiple", 
            type="file",
            label="Upload Images (.png or .jpg)"
        ),   
    outputs=gr.File(file_types=[".png"]),
    title="Segmentation Inference Demo",
    description="Upload multiple images. Predicted masks will be saved in output_imgs and downloadable."
)

demo.launch(share = True)

# Check model sizes
fp32_size = os.path.getsize("models/resunet.onnx") / (1024**2)
fp16_size = os.path.getsize("models/quantized_model.onnx") / (1024**2)

print(f"\nModel sizes:")
print(f"FP32: {fp32_size:.2f} MB")
print(f"FP16: {fp16_size:.2f} MB")
print(f"Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%")