import argparse
import yaml
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.data.splits import loso_split
from src.models import get_model

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_fold(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm

def main(config_path):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs, f1s = [], []
    total_cm = np.zeros((5, 5), dtype=int)  # 5 classes: Hello, Help me, Stop, Thank you, Yes

    for train_ids, test_id in loso_split(cfg["data"]["preprocessed_dir"]):
        print(f"=== Evaluating fold: Test {test_id} ===")

        # dataset
        test_set = EEGDataset(cfg["data"]["preprocessed_dir"], subject_ids=[test_id])
        test_loader = DataLoader(test_set, batch_size=cfg["training"]["batch_size"])

        # model
        model = get_model(cfg["model"],
                          n_channels=len(cfg["preprocessing"]["channels"]),
                          n_classes=5,
                          input_samples=test_set.trials.shape[-1])
        model = model.to(device)

        # load correct checkpoint
        ckpt_dir = cfg["results"]["checkpoint_dir"]
        ckpt_path = os.path.join(ckpt_dir, f"{test_id}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for {test_id}: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # evaluate
        acc, f1, cm = evaluate_fold(model, test_loader, device)
        print(f"Subject {test_id} | Acc: {acc:.3f} | F1: {f1:.3f}")
        accs.append(acc)
        f1s.append(f1)
        total_cm += cm

    # overall
    print("\n=== Overall Results ===")
    print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean F1-score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print("Confusion Matrix (all folds):")
    print(total_cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
