import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset import prepare_dataloaders
from model import DocumentCNN

def evaluate_baseline(test_loader):
    """Majority Predictor Baseline"""
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    majority_class = np.bincount(all_labels).argmax()
    baseline_preds = np.full_like(all_labels, fill_value=majority_class)

    acc = accuracy_score(all_labels, baseline_preds)
    f1 = f1_score(all_labels, baseline_preds, average='macro', zero_division=0)
    return acc, f1, majority_class

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    failures = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                p = preds[i].item()
                l = labels[i].item()
                all_preds.append(p)
                all_labels.append(l)

                if p != l and len(failures) < 5:
                    failures.append({"true": l, "pred": p})

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1, all_labels, all_preds, failures

def main():
    parser = argparse.ArgumentParser(description="Evaluate DocumentCNN on RVL-CDIP Test Set")
    parser.add_argument("--data_path", type=str, required=True, help="Path to RVL-CDIP dataset")
    parser.add_argument("--model_path", type=str, default="./best_model.pth", help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = prepare_dataloaders(args.data_path, batch_size=64, seed=args.seed)

    # 1. Baseline Evaluation
    base_acc, base_f1, maj_cls = evaluate_baseline(test_loader)
    print("=" * 50)
    print(f"Majority Predictor Baseline (Class: {classes[maj_cls]})")
    print(f"Accuracy: {base_acc*100:.2f}% | Macro F1: {base_f1:.4f}")
    print("=" * 50)

    # 2. Trained Model Evaluation
    model = DocumentCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    acc, f1, targets, preds, failures = evaluate_model(model, test_loader, device)
    print("\nTrained DocumentCNN Results:")
    print(f"Test Accuracy: {acc*100:.2f}% | Test Macro F1: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(targets, preds, target_names=classes, zero_division=0))

    print("Sample Failure Cases (for Q2/Q3 report):")
    for idx, f in enumerate(failures, 1):
        print(f"  {idx}. True Class: '{classes[f['true']]}' | Predicted: '{classes[f['pred']]}'")

if __name__ == "__main__":
    main()