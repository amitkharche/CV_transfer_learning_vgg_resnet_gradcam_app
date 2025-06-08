import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocessing.custom_loader import get_data_loaders
from src.models.vgg_resnet import get_vgg_model, get_resnet_model
from src.models.vision_transformer import get_vit_model
import argparse
import os

def train(model_type='vgg', data_dir='data/', save_path='output/model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = get_data_loaders(data_dir=data_dir)

    if model_type == 'vgg':
        model = get_vgg_model(num_classes)
    elif model_type == 'resnet':
        model = get_resnet_model(num_classes)
    elif model_type == 'vit':
        model = get_vit_model()
    else:
        raise ValueError("Unsupported model type")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):  # You can increase epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if model_type == 'vit':
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["vgg", "resnet", "vit"], required=True)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--save_path", default="output/model.pth")
    args = parser.parse_args()
    train(args.model_type, args.data_dir, args.save_path)
