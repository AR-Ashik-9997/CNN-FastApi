import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_binary_filter():

    model_path = MODELS_DIR / "load_binary_filter_of_brain_mri_model.pth"
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

binary_filter_model = load_binary_filter()

def verify_with_binary_model(image: Image.Image) -> bool:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = binary_filter_model(tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()

    if predicted_class == 0 and confidence > 0.50:
        return True
    return False