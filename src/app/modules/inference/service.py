import io
import torch
import numpy as np
from typing import List
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


def image_process(files: List[UploadFile]) -> torch.Tensor:
    try:
        tensors = []
        for file in files:
            file_bytes = file.file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            tensor = eval_transform(image)
            tensors.append(tensor)
        batch_tensor = torch.stack(tensors)
        return batch_tensor
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Image preprocessing failed: {str(e)}"
        )


def run_cnn_inference(files: List[UploadFile]) -> List[CNNInferenceResponse]:
    model = ModelManager.get_model("brain_tumor_model")
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")

    input_tensor = image_process(files)
    outputs = None

    try:
        if hasattr(model, "eval") and callable(model):
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                tensor_input = input_tensor.to(device)

                raw_out = model(tensor_input)
                probs = torch.softmax(raw_out, dim=1)
                outputs = probs.cpu().numpy()
        else:
            raise HTTPException(
                status_code=500, detail="Unsupported DL model framework."
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model execution error: {str(e)}")

    class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
    response_list = []

    for output in outputs:
        predicted_class = int(np.argmax(output))
        confidence = f"{float(output[predicted_class]) * 100:.2f}%"
        class_label = class_names[predicted_class]
        all_probs = [float(p) for p in output]

        response_list.append(
            CNNInferenceResponse(
                predicted_class=predicted_class,
                class_label=class_label,
                confidence=confidence,
                all_probabilities=all_probs,
            )
        )
    return response_list
