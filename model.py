import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(DocumentCNN, self).__init__()
        
        # Block 1: Input (1, 64, 64) -> Output (32, 32, 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: Input (32, 32, 32) -> Output (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: Input (64, 16, 16) -> Output (128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        # 💥 แก้ไขตรงนี้: 64x64 รูปจะถูกลดขนาดลง 3 รอบ (หาร 2 สามครั้ง) เหลือขนาด 8x8 
        # Feature channels = 128 -> 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x