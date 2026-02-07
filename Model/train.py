import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from datasets import load_dataset

# --- 1. Model Definition ---
class MyCNN(nn.Module):
    def __init__(self, in_channel, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. Data Preprocessing & Loading ---
def get_dataloaders(batch_size=128):
    dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
    norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    val_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    def preprocess(ex, trans):
        ex['pixel_values'] = [trans(img.convert("RGB")) for img in ex['image']]
        return ex

    split = dataset['train'].train_test_split(test_size=0.2)
    train_ds = split['train'].with_transform(lambda x: preprocess(x, train_trans))
    val_ds = split['test'].with_transform(lambda x: preprocess(x, val_trans))

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]).float().unsqueeze(1)
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader

# --- 3. Training Logic ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    correct, total = 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()

        with autocast('cuda'):
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (logits > 0.0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total)*100:.2f}%"})

# --- 4. Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()
    
    model = MyCNN(in_channel=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    for epoch in range(1, 6):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"model_epoch_{epoch}.pth")
