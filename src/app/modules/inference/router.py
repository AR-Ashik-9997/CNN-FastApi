from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.modules.inference.schemas import CNNInferenceResponse
from app.modules.inference import service

router = APIRouter(prefix="/inference", tags=["AI/ML Inference Engine"])


@router.post("/predict/images", response_model=List[CNNInferenceResponse])
async def predict_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded.",
        )

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' must be a valid image format (JPG, PNG, JPEG).",
            )

    return service.run_cnn_inference(files)
