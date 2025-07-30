import pandas as pd
from PIL import Image 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import os 

# Define a custom dataset 
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Check if image directory exists
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir 
        self.transform = transform 

        # Check if required columns exist
        required_columns = ['Finding Labels', 'Image Index']
        for col in required_columns:
            if col not in self.labels_df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")

        # Create a mapping for class labels to integers 
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_df['Finding Labels'].unique())}

    def __len__(self): 
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx]['Image Index'])
        
        # Check if image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")
            
        label = self.label_to_idx[self.labels_df.iloc[idx]['Finding Labels']]

        if self.transform: 
            image = self.transform(image)

        return image, label 
    
# Define transform 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and loader 
try:
    dataset = CustomImageDataset(csv_file='C:\\Users\\DELL\\Documents\\Test\\meta.csv', img_dir = 'C:\\Users\\DELL\\Documents\\Test\\Img', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f'Total samples: {len(dataset)}')

    image, label = dataset[0]
    print(f'Image shape: {image.shape}')
    print(f'Label index: {label}')

    for i in range(6):
        img, lbl = dataset[i]
        print(f'Sample {i}: label = {lbl}, image shape = {img.shape}')
        
except FileNotFoundError as e:
    print(f"File not found error: {e}")
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")