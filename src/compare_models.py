import matplotlib.pyplot as plt

def compare_models(results):
    names = [x['name'] for x in results]
    accs = [x['accuracy'] for x in results]
    plt.figure(figsize=(8, 5))
    plt.bar(names, accs, color='skyblue')
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("output/model_comparison.png")
    plt.show()

if __name__ == "__main__":
    sample_results = [
        {"name": "VGG", "accuracy": 0.88},
        {"name": "ResNet", "accuracy": 0.91},
        {"name": "ViT", "accuracy": 0.89}
    ]
    compare_models(sample_results)
