from transformers import ViTForImageClassification

def get_vit_model(num_labels=2):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # ✅ THIS FIXES THE ERROR
    )
    return model
