from data import TrainDataset, TestDataset
from model import CNN  # 假设有多个模型类，如 CNN, ResNet, etc.
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------
# Configurations
# -------------------------
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8

# 模型配置字典
MODEL_CONFIGS = {
    "CNN": {"model": CNN, "lr": 0.001},
    # 如果有其他模型，可以在这里添加
    # "ResNet": {"model": ResNet, "lr": 0.0001},
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for batch in tqdm(loader, desc="Train"):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy

def train_and_evaluate(model_name, model_class, lr, device):
    print(f"Training model: {model_name}")
    train_ds = TrainDataset()
    test_ds = TestDataset()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = model_class().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = {"model": model_name, "epochs": []}

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        results["epochs"].append({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})

    return results

import json

def main():
    all_results = []
    for model_name, config in MODEL_CONFIGS.items():
        model_class = config["model"]
        lr = config["lr"]
        results = train_and_evaluate(model_name, model_class, lr, DEVICE)
        all_results.append(results)

    # 打印最终结果
    print("\nFinal Results:")
    for result in all_results:
        print(f"Model: {result['model']}")
        for epoch_result in result["epochs"]:
            print(f"  Epoch {epoch_result['epoch']} - Val Loss: {epoch_result['val_loss']:.4f}, Val Accuracy: {epoch_result['val_accuracy']:.4f}")

    # 将结果存储到 JSON 文件
    results_file = "UCAS_NLP_25_Fall/work1/code/training_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n训练结果已保存到文件: {results_file}")

if __name__ == "__main__":
    main()