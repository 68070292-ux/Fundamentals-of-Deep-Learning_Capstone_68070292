import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentCNN(nn.Module):
    """
    Custom CNN Architecture สำหรับ Document Image Classification
    ใช้องค์ประกอบจากวิชา: Conv2d, BatchNorm, MaxPool, Dropout, Linear (FC)
    """
    def __init__(self, num_classes=16):
        super(DocumentCNN, self).__init__()
        
        # Block 1: Input (1 x 128 x 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Classifier (128 channels * 16 x 16 spatial size หลังผ่าน pool 3 ครั้ง)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 128 -> 64
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 64 -> 32
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 32 -> 16
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x