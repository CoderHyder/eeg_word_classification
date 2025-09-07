import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.data.splits import loso_split
from src.models import get_model

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_one_fold(model, train_loader, val_loader, cfg, device, test_id):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"])
    
    best_val_acc = 0
    patience = 0
    
    for epoch in range(cfg["training"]["num_epochs"]):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total if total > 0 else 0
        
        # --- validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']} "
              f"| Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
        # early stopping + save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            ckpt_dir = cfg["results"]["checkpoint_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{test_id}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping")
                break
    
    return best_val_acc

def main(config_path):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LOSO splits
    for train_ids, test_id in loso_split(cfg["data"]["preprocessed_dir"]):
        print(f"=== Fold: Test {test_id} ===")
        train_set = EEGDataset(cfg["data"]["preprocessed_dir"], subject_ids=train_ids)
        test_set = EEGDataset(cfg["data"]["preprocessed_dir"], subject_ids=[test_id])
        
        train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg["training"]["batch_size"])
        
        # build model
        model = get_model(cfg["model"],
                          n_channels=len(cfg["preprocessing"]["channels"]),
                          n_classes=5,
                          input_samples=train_set.trials.shape[-1])
        model = model.to(device)
        
        # train one fold
        best_acc = train_one_fold(model, train_loader, test_loader, cfg, device, test_id)
        print(f"Best Val Acc for subject {test_id}: {best_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
