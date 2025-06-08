from transformers import ViTForImageClassification

def get_vit_model():
    return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2)
