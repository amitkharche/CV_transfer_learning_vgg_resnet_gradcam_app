import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from src.models.vgg_resnet import get_vgg_model, get_resnet_model
from src.explainability.grad_cam import RobustGradCAM as GradCAM
import numpy as np
import cv2

st.set_page_config(page_title="Vision Transfer Learning Demo")
st.title("🔍 Vision Transfer Learning Demo (VGG16 & ResNet50)")

# ✅ Replace all inplace=True ReLU with inplace=False
def replace_relu_with_non_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_non_inplace(child)

# Model selection dropdown
model_type = st.selectbox("Choose model", ["vgg", "resnet"])

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare the selected model
    if model_type == "vgg":
        model = get_vgg_model()
        model.load_state_dict(torch.load("output/vgg_model.pth", map_location=device))
        replace_relu_with_non_inplace(model)  # ✅ Critical Fix
        model.to(device).eval()
        # VGG16 — last Conv2d before classifier
        target_layer = model.features[28]
        cam_generator = GradCAM(model, target_layer)

    elif model_type == "resnet":
        model = get_resnet_model()
        model.load_state_dict(torch.load("output/resnet_model.pth", map_location=device))
        replace_relu_with_non_inplace(model)  # ✅ Critical Fix
        model.to(device).eval()
        # ResNet50 — last bottleneck block's conv3
        target_layer = model.layer4[-1].conv3
        cam_generator = GradCAM(model, target_layer)

    # Predict class
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        st.success(f"Predicted Class: {pred}")

    # Generate Grad-CAM heatmap
    try:
        cam = cam_generator.generate(input_tensor)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = np.array(image.resize((224, 224)))
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        st.image(superimposed, caption="Grad-CAM", use_container_width=True)
    except Exception as e:
        st.error(f"Grad-CAM generation failed: {e}")
        import traceback
        st.error(traceback.format_exc())
