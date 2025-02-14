import torch
from torch.utils.data import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns 'image' and 'label'.
        """
        self.df = dataframe
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract image and label from the DataFrame
        row = self.df.iloc[idx]
        img = row['image']   # This might be a numpy array or something similar
        label = row['label'] # Integer label
        
        # Ensure img is a numpy array:
        img_array = np.array(img, dtype=np.float32)
        
        # Preprocess image: pad to 32x32 if originally 28x28
        # Initialize a 32x32 zero array
        img_padded = np.zeros((32,32), dtype=np.float32)
        img_padded[2:30, 2:30] = img_array
        
        # Normalize (if desired)
        img_padded /= 255.0
        
        # Add a channel dimension: (1,32,32)
        img_tensor = torch.from_numpy(img_padded).unsqueeze(0)  # shape: [1, 32, 32]

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
