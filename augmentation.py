
from imports import *
 
def get_test_augmentation():
    """
    Comprehensive augmentations for retinal vessel segmentation:
    - Geometric: Rotation, Flip, Elastic Transform
    - Color: Brightness, Contrast, Gamma, Hue/Saturation
    - Noise: Gaussian Noise, Blur
    - Advanced: GridDistortion, OpticalDistortion
    """
    return A.Compose([
        A.Resize(512, 512),
       A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])