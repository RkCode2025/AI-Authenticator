import torch
from train import MyCNN, get_dataloaders

def evaluate_model(weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import dataloaders from train.py logic
    _, val_loader = get_dataloaders(batch_size=64)
    
    # Initialize Model
    model = MyCNN(in_channel=3).to(device)
    
    # Load Saved Weights
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    correct, total = 0, 0
    
    print(f"Starting evaluation on {device}...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(inputs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    print("-" * 30)
    print(f"FINAL TEST ACCURACY: {(correct/total)*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Ensure this filename matches your saved checkpoint
    evaluate_model("model_epoch_5.pth")
