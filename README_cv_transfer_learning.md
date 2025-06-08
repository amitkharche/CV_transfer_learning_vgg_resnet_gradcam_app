# 🔍 Project 6: Transfer Learning for Vision (`cv-transfer-learning`)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Torch](https://img.shields.io/badge/PyTorch-1.12+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Business Problem

In computer vision, creating high-performing models with limited data is a common challenge. Transfer learning allows leveraging pretrained models to significantly reduce training time and increase performance.

This project demonstrates how to use:
- ✅ Pretrained **VGG** and **ResNet** on a **custom image dataset**
- ✅ **Vision Transformer (ViT)** from Hugging Face
- ✅ **Grad-CAM** for model explainability
- ✅ A **Streamlit app** for real-time image prediction

---

## 📁 Dataset Structure

You must organize your dataset as follows:

```
data/
├── train/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

Example (Binary classification):
```
data/train/dogs/
data/train/cats/
data/val/dogs/
data/val/cats/
```

Images should be in JPG/PNG format.

---

## 🗂️ Project Structure

```
cv-transfer-learning/
├── data/                         # Custom dataset (not included)
├── notebooks/                    # Future Jupyter notebooks
├── output/                       # Trained model weights and plots
├── src/
│   ├── models/
│   │   ├── vgg_resnet.py         # Load pretrained VGG and ResNet
│   │   └── vision_transformer.py # Load Hugging Face ViT
│   ├── preprocessing/
│   │   └── custom_loader.py      # ImageFolder-based data loader
│   ├── explainability/
│   │   └── grad_cam.py           # Grad-CAM visual explanation
│   ├── utils/
│   │   ├── metrics.py            # Accuracy/F1 utilities
│   │   └── timer.py              # Elapsed time tracker
│   ├── train.py                  # Train CLI
│   ├── evaluate.py               # Evaluate CLI
│   └── compare_models.py         # Performance comparison
├── streamlit_app/
│   └── app.py                    # Streamlit app for prediction + Grad-CAM
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # You're here!
```

---

## 🛠️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 📁 Prepare Dataset

Organize your dataset as per the structure above under `/data`.

---

## 🚀 Training

Train a model (VGG, ResNet, or ViT):

```bash
python src/train.py --model_type vgg --data_dir data/ --save_path output/vgg_model.pth
python src/train.py --model_type resnet --data_dir data/ --save_path output/resnet_model.pth
python src/train.py --model_type vit --data_dir data/ --save_path output/vit_model.pth
```

---

## 📈 Evaluation

Evaluate a trained model on validation data:

```bash
python src/evaluate.py --model_type vgg --data_dir data/ --model_path output/vgg_model.pth
```

---

## 📊 Model Comparison

Generate a simple bar chart to compare models:

```bash
python src/compare_models.py
```

Edit the `sample_results` dictionary inside the script to reflect your actual model performances.

---

## 🖼️ Streamlit App – Image Upload & Prediction

```bash
streamlit run streamlit_app/app.py
```

- Upload an image
- Choose a model (VGG, ResNet, ViT)
- View predicted class and Grad-CAM heatmap

---

## 🔍 Explainability

**Grad-CAM** is implemented for:
- ✅ VGG
- ✅ ResNet

Vision Transformers currently do not support Grad-CAM in this version.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For feedback or collaboration, connect via [LinkedIn](https://www.linkedin.com).
