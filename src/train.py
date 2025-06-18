import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing.custom_loader import get_data_loaders
from src.models.vgg_resnet import get_vgg_model, get_resnet_model
from src.models.vision_transformer import get_vit_model

import argparse

def train(model_type='vgg', data_dir='data/', save_path='output/model.pth', epochs=3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load dataset
    print("📥 Loading data...")
    train_loader, val_loader, num_classes = get_data_loaders(data_dir=data_dir, batch_size=batch_size)

    # Select model
    print(f"🧠 Initializing model: {model_type.upper()}")
    if model_type == 'vgg':
        model = get_vgg_model(num_classes)
    elif model_type == 'resnet':
        model = get_resnet_model(num_classes)
    elif model_type == 'vit':
        model = get_vit_model()
    else:
        raise ValueError("Unsupported model type. Choose from ['vgg', 'resnet', 'vit'].")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Starting training...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if model_type == 'vit':
                outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Model using Transfer Learning")
    parser.add_argument("--model_type", choices=["vgg", "resnet", "vit"], required=True, help="Choose model type")
    parser.add_argument("--data_dir", default="data/", help="Path to data directory")
    parser.add_argument("--save_path", default="output/model.pth", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()
    train(args.model_type, args.data_dir, args.save_path, args.epochs, args.batch_size)
