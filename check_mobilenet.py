import torch
import torchvision.models as models

model = models.mobilenet_v3_large(weights=None)
print("MobileNetV3 classifier structure:")
print(model.classifier)
print("\nIndices:")
for i, layer in enumerate(model.classifier):
    print(f"  [{i}]: {layer}")

# Check the last linear layer
print(f"\nLast layer (classifier[-1]): {model.classifier[-1]}")
print(f"Output size: {model.classifier[-1].out_features}")
