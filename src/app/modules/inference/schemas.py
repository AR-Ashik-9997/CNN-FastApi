from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class CNNInferenceResponse(BaseModel):
    predicted_class: int = Field(..., description="Predicted class index")
    class_label: Optional[str] = Field(default=None, description="Human readable label")
    confidence: str = Field(..., description="Probability")
    # all_probabilities: List[float] = Field(
    #     default=None, description="Softmax probabilities for all classes"
    # )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": 0,
                "class_label": "Glioma Tumor",
                "confidence": 0.965,
                # "all_probabilities": [0.965, 0.020, 0.010, 0.005],
            }
        }
    )
