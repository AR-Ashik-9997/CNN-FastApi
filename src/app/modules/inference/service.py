import io
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from fastapi import UploadFile, HTTPException
from app.core.ml_loader import ModelManager
from app.modules.inference.schemas import CNNInferenceResponse

eval_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def image_process(file_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = eval_transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Image preprocessing failed: {str(e)}"
        )


def run_cnn_inference(file: UploadFile) -> CNNInferenceResponse:
    model = ModelManager.get_model("brain_tumor_model")
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")

    file_bytes = file.file.read()
    input_tensor = image_process(file_bytes)
    outputs = None

    try:
        if hasattr(model, "eval") and callable(model):
            model.eval()
            with torch.no_grad():

                device = next(model.parameters()).device
                tensor_input = input_tensor.to(device)

                raw_out = model(tensor_input)
                probs = torch.softmax(raw_out, dim=1)
                outputs = probs.cpu().numpy()[0]
        else:
            raise HTTPException(
                status_code=500, detail="Unsupported DL model framework."
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model execution error: {str(e)}")

    class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
    predicted_class = int(np.argmax(outputs))
    confidence = float(outputs[predicted_class])
    class_label = class_names[predicted_class]
    all_probs = [float(p) for p in outputs]

    return CNNInferenceResponse(
        predicted_class=predicted_class,
        class_label=class_label,
        confidence=confidence,
        all_probabilities=all_probs,
    )
