import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ArtBenchDataset(Dataset):
    def __init__(self, data_root, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        batch_folder = os.path.join(data_root, "artbench-10-python", "artbench-10-batches-py")
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
            
        for file_name in files:
            file_path = os.path.join(batch_folder, file_name)
            if not os.path.exists(file_path): continue
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                # Handle potential bytes key
                labels_key = 'labels' if 'labels' in entry else b'labels'
                self.labels.extend(entry[labels_key])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloader(data_root, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ArtBenchDataset(data_root, train=train, transform=transform)
    
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train, 
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=True if use_cuda else False
    )
