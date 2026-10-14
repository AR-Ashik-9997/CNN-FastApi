# CNN FastAPI - Brain Tumor MRI Classification API

A production-ready FastAPI service for classifying brain MRI scans using a convolutional neural network (CNN). The application validates uploaded medical images, filters invalid MRI inputs with a binary classification model, and predicts one of four tumor categories.

## Overview

This project exposes a REST API that accepts MRI images and returns:

- the predicted tumor class
- the human-readable class label
- the model confidence as a percentage

The current implementation classifies images into the following categories:

1. Glioma Tumor
2. Meningioma Tumor
3. No Tumor
4. Pituitary Tumor

## Features

- FastAPI-based REST API with CORS enabled
- Multi-image upload support
- Brain MRI validation using a binary verification model
- CNN-based image classification using PyTorch
- Model loading on startup for low-latency inference
- Standardized API response schema using Pydantic

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- PyTorch
- TorchVision
- Pillow
- NumPy
- Pydantic
- Joblib

## Project Structure

```text
CNN-FastApi/
├── src/
│   ├── app/
│   │   ├── cnn_model/
│   │   │   └── simpleCNN.py
│   │   ├── core/
│   │   │   ├── binary_filter.py
│   │   │   ├── config.py
│   │   │   ├── __init__.py
│   │   │   └── ml_loader.py
│   │   ├── modules/
│   │   │   └── inference/
│   │   │       ├── router.py
│   │   │       ├── schemas.py
│   │   │       ├── service.py
│   │   │       └── __init__.py
│   │   ├── __init__.py
│   │   └── main.py
│   └── models/
│       ├── brain_tumor_model.pkl
│       └── load_binary_filter_of_brain_mri_model.pth
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Model Details

The application uses two trained models:

- Binary MRI filter: verifies whether an uploaded image is a valid brain MRI scan
- CNN classifier: predicts the tumor class from the validated MRI image

### CNN Architecture

The model is implemented in `src/app/cnn_model/simpleCNN.py` and uses a lightweight convolutional network with:

- 3 convolutional blocks
- ReLU activations
- Max pooling layers
- Fully connected classifier head

The classifier predicts four classes after image preprocessing and normalization.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AR-Ashik-9997/CNN-FastApi.git
cd CNN-FastApi
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
```

3. Activate the environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Optional: install the project in editable mode:

```bash
pip install -e .
```

## Environment Setup

The project uses Pydantic settings with `.env` support. The default configuration is defined in `src/app/core/config.py`.

You may create a `.env` file in the project root if you need to override default settings:

```env
PROJECT_NAME="CNN FastAPI - Brain Tumor Detection"
API_V1_STR="/api/v1"
```

## Running the Application

From the project root, set the Python path and start Uvicorn:

```bash
set PYTHONPATH=src
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On macOS/Linux:

```bash
export PYTHONPATH=src
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/api/v1/openapi.json

## API Endpoints

### Root Endpoint

```http
GET /
```

Returns an online status response:

```json
{
  "status": "online",
  "system": "Enterprise AI/ML FastAPI Platform"
}
```

### Predict Brain Tumor from MRI Images

```http
POST /api/v1/inference/predict/images
```

Request type: `multipart/form-data`

Form field:

- `files`: one or more valid image files (`.jpg`, `.jpeg`, `.png`)

#### Example using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/inference/predict/images" \
  -F "files=@sample_mri.jpg"
```

#### Example response

```json
[
  {
    "predicted_class": 0,
    "class_label": "Glioma Tumor",
    "confidence": "96.50%"
  }
]
```

## Request Validation

The service performs the following checks before running inference:

- ensures at least one file is uploaded
- verifies the file is an image
- validates the image through the binary MRI filter model
- preprocesses the image to match the model input size and normalization settings

If any of these checks fail, the API returns an HTTP 400 response with a descriptive error message.

## Notes and Considerations

- The trained model files must be present in the `src/models` directory.
- The system is designed for image-based classification experiments and demo use.
- For medical decision support, this should be validated with domain-specific clinical workflows and proper medical review.
- The app is intentionally lightweight and easy to deploy in local or cloud environments.

## License

This project does not currently declare a formal license in the repository metadata. Use it according to the repository owner’s distribution and licensing policy.

## Contribution

Contributions, improvements, and model enhancements are welcome. For major changes, it is recommended to open an issue or discuss the proposed changes before submitting a pull request.
