import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models.vgg_resnet import get_vgg_model, get_resnet_model
from src.models.vision_transformer import get_vit_model
from src.explainability.grad_cam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title("🔍 Vision Transfer Learning Demo")

model_type = st.selectbox("Choose model", ["vgg", "resnet", "vit"])
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vgg":
        model = get_vgg_model()
        model.load_state_dict(torch.load("output/vgg_model.pth", map_location=device))
        model.to(device).eval()
        cam_generator = GradCAM(model, model.features[-1])
    elif model_type == "resnet":
        model = get_resnet_model()
        model.load_state_dict(torch.load("output/resnet_model.pth", map_location=device))
        model.to(device).eval()
        cam_generator = GradCAM(model, model.layer4[2].conv3)
    else:
        model = get_vit_model()
        model.load_state_dict(torch.load("output/vit_model.pth", map_location=device))
        model.to(device).eval()
        cam_generator = None  # ViT doesn't support Grad-CAM well yet

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        if model_type == "vit":
            pred = torch.argmax(output.logits, dim=1).item()
        else:
            pred = torch.argmax(output, dim=1).item()
        st.success(f"Predicted Class: {pred}")

    if cam_generator:
        cam = cam_generator.generate(input_tensor)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = np.array(image.resize((224, 224)))
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        st.image(superimposed, caption="Grad-CAM", use_column_width=True)
