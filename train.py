import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import prepare_dataloaders
from model import DocumentCNN

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        # อัปเดต Loss/Acc บน Progress Bar
        pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total*100:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train DocumentCNN on RVL-CDIP")
    parser.add_argument("--data_path", type=str, required=True, help="Path to RVL-CDIP dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Data Loaders
    train_loader, val_loader, _, classes = prepare_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # Initialize Model
    model = DocumentCNN(num_classes=len(classes))
    
    # รองรับ Multi-GPU (GPU T4 x2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    
    save_path = os.path.join(args.output_dir, "best_model.pth")

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # บันทึก Weight ของ Epoch ที่ได้ Validation Accuracy สูงที่สุด
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # รองรับการดึง state_dict ทั้งแบบ Single GPU และ Multi-GPU
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"  --> Saved best model checkpoint to {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

if __name__ == "__main__":
    main()