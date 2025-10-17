## Retinal Vessel Segmentation

Retinal vessel segmentation plays a pivotal role in the early detection and management of ocular and systemic diseases, including diabetic retinopathy, glaucoma, hypertension, and cardiovascular conditions. As a result, retinal images have been widely used to detect early signs of systemic vascular disease. In order to facilitate the diagnosis of systemic vascular diseases, vessels need to be accurately segmented. 
By precisely delineating blood vessels in fundus images, this technique enables automated screening for vascular abnormalities, potentially reducing diagnostic delays and improving patient outcomes in resource-constrained clinical settings.

### Approach

This project introduces ResUNet Deeper, an enhanced U-Net architecture incorporating residual blocks for superior feature extraction and gradient flow. Trained from scratch on the FIVES (Fundus Images for Vessel Segmentation) dataset, the model achieves robust performance and boost generalization through CLAHE enhancement on all RGB channels with circular FOV masking for robust vessel contrast, 11 diverse augmentations, and a custom weighted loss function. To facilitate deployment on edge devices, the model is exported to ONNX format and quantized to FP16, yielding a 50% size reduction and 2x inference speedup without accuracy degradation.

### Dataset 

ResUnet Model trained from scratch on [Fundus Images.](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation/data)

Model weights are available on [HuggingFace](https://huggingface.co/ramshakhan/segmentation-model/tree/main)

### Onnx Model graph 

https://github.com/user-attachments/assets/52a8102f-d03b-47b7-9d94-a4443b7d5bac

### Model Architecture

<img width="1238" height="882" alt="Screenshot 2025-10-17 141230" src="https://github.com/user-attachments/assets/6258e05a-2d05-43a0-9fe6-8632ea308fcf" />

### Loss curve

<img width="807" height="520" alt="Screenshot 2025-10-15 120319" src="https://github.com/user-attachments/assets/cf368153-ada2-4449-af76-2df009414be2" />

### Dice score

<img width="805" height="514" alt="Screenshot 2025-10-15 120328" src="https://github.com/user-attachments/assets/b5edbe07-1107-470b-a755-51a444a91092" />

### IoU

<img width="800" height="521" alt="Screenshot 2025-10-15 120334" src="https://github.com/user-attachments/assets/fcd9325c-6586-45b1-9641-f9932b22b0c0" />

### Performance Metrics

| Dataset | Loss | IoU | Dice | Accuracy |   
|----------|------|------|------|-----------|
| **Validation** | 0.082 | 0.796 | 0.886 | 0.984 | 
| **Test** | 0.082 | 0.797 | 0.887 | 0.984 |
| **Cross-dataset (DRIVE)** | 0.164 | 0.637 | 0.778 | 0.962 | 

> **Note:** The model shows stable in-domain performance (val/test Dice: 0.89 on FIVES, no overfitting). On cross-dataset (DRIVE), it maintains 96.2% accuracy despite a ~12% Dice drop from fundus acquisition shifts - demonstrating residual blocks, preprocessing and augmentation's role in generalization.
> 
## Inference

- **Format**: PyTorch → ONNX (FP16 quantized)
- **Optimization**: ~50.2% model size reduction (from 243 MB to 121 MB), 2× inference speedup
- **Compatibility**: Edge device ready, no accuracy degradation

## Loss Function

#### Weighted Combined Loss = 0.5 * DiceLoss + 0.3 * Binary Cross Entropy Loss + 0.2 * Focal Loss





