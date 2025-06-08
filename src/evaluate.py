import torch
from sklearn.metrics import classification_report
from src.preprocessing.custom_loader import get_data_loaders
from src.models.vgg_resnet import get_vgg_model, get_resnet_model
from src.models.vision_transformer import get_vit_model
import argparse

def evaluate(model_type='vgg', data_dir='data/', model_path='output/model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes = get_data_loaders(data_dir=data_dir)

    if model_type == 'vgg':
        model = get_vgg_model(num_classes)
    elif model_type == 'resnet':
        model = get_resnet_model(num_classes)
    elif model_type == 'vit':
        model = get_vit_model()
    else:
        raise ValueError("Unsupported model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            if model_type == 'vit':
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["vgg", "resnet", "vit"], required=True)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--model_path", default="output/model.pth")
    args = parser.parse_args()
    evaluate(args.model_type, args.data_dir, args.model_path)
