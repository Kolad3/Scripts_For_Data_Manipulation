import pandas as pd
from PIL import Image 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import os 

# Define a custom dataset 
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir 
        self.transform = transform 

        # Create a mapping for class labels to integers 
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_df['label'].unique())}

    def __len__(self): 
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_idx[self.labels_df.iloc[idx]['label']]

        if self.transform: 
            image = self.transform(image)

        return image, label 
    
# Define transfor 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and loader 
dataset = CustomImageDataset(csv_file='dataset/labels.csv', img_dir = 'dataset/images', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)