import joblib
import torch
import os
from pathlib import Path
from app.cnn_model.simpleCNN import SimpleCNN

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelManager:
    _models = {}

    @classmethod
    def load_all_models(cls):
        if not MODELS_DIR.exists():
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        old_restore = torch.serialization.default_restore_location

        def custom_restore_location(storage, location):
            return old_restore(storage, "cuda" if torch.cuda.is_available() else "cpu")

        torch.serialization.default_restore_location = custom_restore_location

        for file_path in MODELS_DIR.glob("*.pkl"):
            model_name = file_path.stem
            print(f"Loading AI Model '{model_name}' from {file_path}...")
            try:
                if "brain_tumor_model" in model_name:
                    model = SimpleCNN(num_classes=4).to(device)
                    state_dict = torch.load(
                        file_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    cls._models[model_name] = model
                else:
                    cls._models[model_name] = joblib.load(file_path)

            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

        print(f"Successfully loaded models: {list(cls._models.keys())}")

    @classmethod
    def get_model(cls, name: str):
        return cls._models.get(name)
