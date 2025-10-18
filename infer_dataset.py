from imports import *


def preprocess_fundus_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    b, g, r = cv2.split(image)
    
    b = clahe.apply(b)
    g = clahe.apply(g)  
    r = clahe.apply(r)
    
    enhanced = cv2.merge([b, g, r])  # Keep all info
    return enhanced
    

def create_mask_for_circular_fov(image):
    """Create circular mask for fundus FOV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask
    
def crop(img, size):
    width , height = img.size
    #todo: handle case where size is less than width or height
    delta0 = size - width
    delta1 = size - height
    padding = (0, 0, delta0, delta1)  
    return transforms.functional.pad(img, padding, fill=0)

class InferDataset(Dataset):
    def __init__(self, image_paths, transform=None, use_preprocessing=True):
        self.image_paths = image_paths
        self.img_lists = sorted(os.listdir(image_paths))
        
        self.transform = transform
        self.use_preprocessing = use_preprocessing
        print(f"Dataset initialized with {len(self.img_lists)} images")
    
    def __len__(self):
        return len(self.img_lists)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_paths, self.img_lists[idx])).convert('RGB')
        
        # Convert to NumPy
        image = np.array(image)
        
        if image is None:
            raise FileNotFoundError(f"Failed to load image or mask at index {idx}")
        
        # Apply preprocessing
        if self.use_preprocessing:
            # Create FOV mask
            fov_mask = create_mask_for_circular_fov(image)
            
            # Preprocess image (CLAHE enhancement)
            image = preprocess_fundus_image(image)
            
            # Apply FOV mask
            image = cv2.bitwise_and(image, image, mask=fov_mask)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
                
        return image