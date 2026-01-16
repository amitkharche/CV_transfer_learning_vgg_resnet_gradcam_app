
---

# Transfer Learning for Vision (`cv-transfer-learning`)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Torch](https://img.shields.io/badge/PyTorch-1.12+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Business Problem

In computer vision, building high-performing models from limited data is a common challenge. Transfer learning allows us to reuse pretrained CNNs like **VGG16** and **ResNet50**, reducing both training time and computational cost while improving accuracy.

This project demonstrates how to:

* ✅ Train **VGG16** and **ResNet50** on a custom image dataset
* ✅ Visualize model decisions using **Grad-CAM**
* ✅ Deploy predictions via a **Streamlit web app**

---

## 📁 Dataset Structure

Organize your dataset in the following format:

```
data/
├── train/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

🔹 Example for binary classification:

```
data/train/cats/
data/train/dogs/
data/val/cats/
data/val/dogs/
```

✅ Image formats: `.jpg`, `.jpeg`, or `.png`

---

## 🗂️ Project Structure

```
cv-transfer-learning/
├── data/                         # Custom dataset (not included)
├── notebooks/                    # Future Jupyter notebooks
├── output/                       # Trained model weights and visual outputs
├── src/
│   ├── models/
│   │   └── vgg_resnet.py         # Model builders for VGG16 and ResNet50
│   ├── preprocessing/
│   │   └── custom_loader.py      # Data loading pipeline using ImageFolder
│   ├── explainability/
│   │   └── grad_cam.py           # Grad-CAM implementations
│   ├── utils/
│   │   ├── metrics.py            # Accuracy, F1, and other metrics
│   │   └── timer.py              # Training time tracker
│   ├── train.py                  # Train models from CLI
│   ├── evaluate.py               # Evaluate trained models
│   └── compare_models.py         # Compare model performance visually
├── streamlit_app/
│   └── app.py                    # Streamlit interface with Grad-CAM overlay
├── requirements.txt              # All dependencies
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

Organize your dataset as shown above under `/data/train` and `/data/val`.

---

## 🚀 Model Training

Train models using pretrained weights:

```bash
# Train VGG16
python src/train.py --model_type vgg --data_dir data/ --save_path output/vgg_model.pth

# Train ResNet50
python src/train.py --model_type resnet --data_dir data/ --save_path output/resnet_model.pth
```

---

## 📈 Model Evaluation

Evaluate model performance on validation set:

```bash
# Evaluate VGG16
python src/evaluate.py --model_type vgg --data_dir data/ --model_path output/vgg_model.pth

# Evaluate ResNet50
python src/evaluate.py --model_type resnet --data_dir data/ --model_path output/resnet_model.pth
```

---

## 📊 Model Comparison

Visualize accuracy of different models:

```bash
python src/compare_models.py
```

Update the `sample_results` dictionary inside `compare_models.py` with your actual results.

---

## 🖼️ Streamlit App – Image Prediction + Grad-CAM

Launch the web interface:

```bash
streamlit run streamlit_app/app.py
```

Features:

* Upload your own image
* Select a model (VGG or ResNet)
* View predicted class
* See **Grad-CAM heatmap** over the predicted region

---

## 🔍 Explainability with Grad-CAM

**Implemented for:**

* ✅ VGG16
* ✅ ResNet50

This version uses a **robust Grad-CAM implementation** that:

* Avoids in-place ReLU conflicts
* Works across architectures
* Generates interpretable heatmaps to show what influenced model decisions

---

## 📄 License

This project is released under the **MIT License**.
Feel free to fork, extend, or use it in your own work.

---

## 📬 Contact

For collaboration, questions, or feedback:
[LinkedIn](https://www.linkedin.com/in/amitkharche)
[Medium](https://medium.com/@amitkharche)
[GitHub](https://github.com/amitkharche)

---
