import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RVLCDIPDataset(Dataset):
    """Custom Dataset สำหรับโหลดรูปเอกสาร RVL-CDIP"""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L') # RVL-CDIP เป็นภาพ Grayscale
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def get_transforms(img_size=128):
    """
    Data Transforms & Augmentation (ใช้ Regularization สอดคล้องกับวิชาเรียน)
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(5), # Small rotation for doc scans
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_test_transform

def prepare_dataloaders(data_dir, batch_size=32, img_size=128, seed=42):
    """
    อ่านโครงสร้างโฟลเดอร์ แบ่ง Train (80%) / Val (10%) / Test (10%)
    """
    torch.manual_seed(seed)
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_filepaths = []
    all_labels = []

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                all_filepaths.append(os.path.join(cls_dir, fname))
                all_labels.append(class_to_idx[cls_name])

    # Shuffle Indices
    indices = list(range(len(all_filepaths)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(indices), generator=generator).tolist()

    n_total = len(indices)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_paths = [all_filepaths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    val_paths = [all_filepaths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    test_paths = [all_filepaths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    train_transform, val_test_transform = get_transforms(img_size)

    train_ds = RVLCDIPDataset(train_paths, train_labels, transform=train_transform)
    val_ds = RVLCDIPDataset(val_paths, val_labels, transform=val_test_transform)
    test_ds = RVLCDIPDataset(test_paths, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes