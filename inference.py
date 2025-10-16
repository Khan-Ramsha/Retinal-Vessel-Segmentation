""""
Inference using quantized model

"""
from augmentation import *
from dataset import *
from metrics import *
from loss_function import *

transform = get_test_augmentation()
img = "images/input"
mask = "images/ground_truth" #these are ground truths
os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset(img, mask, transform = transform)
data_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.3, focal_weight=0.2)

def test_quantized_onnx(model_path, loader, criterion, device):
    print('Available Execution providers: ',ort.get_available_providers())
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_acc = 0
    infer_time = []
    
    with torch.no_grad():
        for images, masks in loader:  
            images = images.cpu().numpy().astype(np.float16)
            masks = masks.to(device)
            
            start = time.time()
            outputs = ort_session.run([output_name], {input_name: images})[0]
            end = time.time()
            
            output_torch = torch.from_numpy(outputs.astype(np.float32)).to(device)
            infer_time.append(end - start)
            loss = criterion(output_torch, masks)
        
            total_loss += loss.item()
            total_iou += calculate_iou(output_torch, masks)
            total_dice += calculate_dice(output_torch, masks)
            total_acc += calculate_accuracy(output_torch, masks)
            
    avg_time = np.mean(infer_time)  
    
    return (total_loss / len(loader), 
            total_iou / len(loader), 
            total_dice / len(loader),
            total_acc / len(loader), 
            avg_time, 
            outputs)