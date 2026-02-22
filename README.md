# Pneumonia Detection from Chest X-ray Images using Deep Learning

A production-style deep learning mini-project for binary chest X-ray classification.

This repository combines model training (transfer learning with VGG19), evaluation, and a Streamlit web app for real-time inference.

The goal is to classify a chest X-ray image into:
- `NORMAL`
- `PNEUMONIA`

## Project Demo
- Demo video: https://www.linkedin.com/posts/nihal-jaiswal-908b52257_deeplearning-pneumoniadetection-healthcareai-activity-7337569916419485696-3gVd

## Why This Project Is Resume-Worthy
- End-to-end workflow from data pipeline to deployable UI.
- Real medical imaging use case with practical relevance.
- Transfer learning approach using a proven CNN backbone (VGG19).
- Modular code split into notebook training + app inference.
- Uses standard ML tooling (TensorFlow, Keras, Streamlit, OpenCV, PIL).
- Demonstrates model packaging and user-facing deployment.

## Problem Statement
Pneumonia diagnosis from chest X-rays is time-sensitive.
Manual screening can be repetitive and high-volume.
This project explores how deep learning can assist by quickly identifying likely pneumonia cases.

## Objectives
- Build a binary image classifier for chest X-ray diagnosis.
- Use transfer learning to reduce training cost and improve convergence.
- Provide a simple browser-based interface for non-technical users.
- Keep the inference flow reproducible and easy to run locally.

## Solution Summary
This project uses a VGG19 feature extractor (`include_top=False`) and a custom dense head.
The training workflow is implemented in:
- `NoteBook/Pneumonia_Detection_Using_Deep_Learning (2).ipynb`

The deployed inference application is implemented in:
- `app.py`

The model checkpoint used by the app is expected at:
- `vgg19_model_01.h5`

## Current Repository Structure

```text
PNeumonia_Project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- kaggle.json
|-- vgg19_model_01.h5
`-- NoteBook/
    `-- Pneumonia_Detection_Using_Deep_Learning (2).ipynb
```

## File-by-File Explanation
- `app.py`
  - Streamlit app entry point.
  - Rebuilds the exact VGG19-based architecture.
  - Loads trained weights and performs image inference.

- `requirements.txt`
  - Python dependency versions for consistent local setup.

- `NoteBook/Pneumonia_Detection_Using_Deep_Learning (2).ipynb`
  - Data loading from Kaggle dataset.
  - Data augmentation setup.
  - Model definition, compile, train, and evaluate.
  - Save/load model weights.

- `kaggle.json`
  - Kaggle API credential file used in notebook workflow.

- `vgg19_model_01.h5`
  - Local model checkpoint for inference.
  - Approx size in this workspace: `424,733,952` bytes (~405 MB).

## Dataset
This project is based on the Kaggle chest X-ray dataset:
- https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Notebook pipeline uses folder-based loading:
- Train: `/chest_xray/train`
- Validation: `/chest_xray/val`
- Test: `/chest_xray/test`

Observed class mapping from notebook:
- `NORMAL: 0`
- `PNEUMONIA: 1`

Observed sample counts from notebook output:
- Train: `5216` images
- Validation: `16` images
- Test: `624` images

## Data Preprocessing and Augmentation
From the training notebook:
- Rescaling: `1./255`
- Input size: `128 x 128`
- Channels: RGB (`(128, 128, 3)`)
- Augmentation configured in training generator includes:
  - Horizontal/vertical flip settings
  - Shear
  - Width shift
  - Height shift
  - Nearest fill mode

For app inference (`app.py`):
- Load image with PIL.
- Convert to RGB.
- Resize to `128 x 128`.
- Normalize by dividing by `255.0`.
- Expand batch dimension.
- Predict via softmax output.

## Model Architecture
Backbone:
- `VGG19`
- `include_top=False`
- `weights='imagenet'`
- Base layers set to non-trainable in notebook setup.

Custom classifier head:
- `Flatten`
- `Dense(4608, activation='relu')`
- `Dropout(0.2)`
- `Dense(1152, activation='relu')`
- `Dense(2, activation='softmax')`

Loss and optimizer (notebook run):
- Loss: `categorical_crossentropy`
- Optimizer: `SGD(learning_rate=1e-4, momentum=0.1, nesterov=True)`
- Metric: `accuracy`

Callbacks used:
- `EarlyStopping` on `val_loss`
- `ModelCheckpoint` on `val_loss`
- `ReduceLROnPlateau` on `val_accuracy`

## Training Snapshot (Current Notebook Run)
The notebook currently shows a short demonstration-style run:
- Epochs run: `1`
- Steps per epoch: `50`

Logged result snippet:
- Train accuracy: `0.7409`
- Train loss: `0.6011`
- Validation accuracy: `0.5000`
- Validation loss: `0.8208`

Evaluation snapshot after loading weights:
- Validation accuracy: `0.5000`
- Validation loss: `0.8208`
- Test accuracy: `0.6308`
- Test loss: `0.6552`

Note:
- These values are from the currently logged notebook execution.
- They represent a single captured run and not final production benchmarking.

## End-to-End Workflow
1. Download chest X-ray dataset via Kaggle API.
2. Create train/val/test generators.
3. Build VGG19 transfer learning model.
4. Compile with SGD and train.
5. Save weights checkpoint.
6. Rebuild same architecture in `app.py`.
7. Load `vgg19_model_01.h5` in Streamlit app.
8. Accept user-uploaded image and return prediction.

## Local Setup Instructions

### 1) Clone the Repository
```bash
git clone <your-repository-url>
cd PNeumonia_Project
```

### 2) Create Virtual Environment
```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

Linux/Mac:
```bash
source .venv/bin/activate
```

### 3) Install Dependencies
```bash
pip install -r requirements.txt
```

### 4) Ensure Model Weights Are Present
Place the trained checkpoint file at project root:
- `vgg19_model_01.h5`

If missing, the app will fail at weight loading step.

### 5) Run Streamlit App
```bash
streamlit run app.py
```

Default app URL:
- `http://localhost:8501`

## How to Use the App
1. Open the Streamlit URL in browser.
2. Upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`).
3. Wait for model inference.
4. Read predicted class:
   - `Normal`
   - `Pneumonia`

## Re-Training Notes
If you want to retrain:
- Open notebook: `NoteBook/Pneumonia_Detection_Using_Deep_Learning (2).ipynb`
- Configure Kaggle credentials using `kaggle.json`.
- Run all cells to reproduce data loading, training, and evaluation.
- Export updated model weights.
- Replace `vgg19_model_01.h5` used by `app.py`.

## Engineering Strengths Demonstrated
- Transfer learning integration for medical image task.
- Practical model deployment using Streamlit.
- Separation of experimentation and inference code paths.
- Reproducible dependency pinning.
- Clear label mapping and directory-based data pipeline.

## Known Limitations
- Current notebook snapshot shows limited epochs.
- Validation split in captured run is very small (`16` images).
- No calibration, uncertainty score, or explainability overlay yet.
- No Docker packaging or CI pipeline yet.
- No automated unit tests for preprocessing/inference path yet.

## Suggested Improvements
- Train for more epochs with stronger validation strategy.
- Use larger and stratified validation split.
- Add Grad-CAM visual explanations for trust and interpretability.
- Add confusion matrix and class-wise metrics.
- Add threshold tuning for recall-sensitive screening.
- Add Docker support for one-command deployment.
- Add API layer (FastAPI/Flask) for service integration.
- Add automated tests and model/version metadata.

## Resume Bullet Ideas (Ready to Use)
- Built an end-to-end deep learning pipeline to classify chest X-ray images into Normal vs Pneumonia using transfer learning with VGG19.
- Developed and deployed a Streamlit-based inference app for real-time medical image prediction with user file upload workflow.
- Implemented data preprocessing and augmentation pipeline using Keras `ImageDataGenerator` and folder-structured datasets.
- Designed custom classifier head over ImageNet-pretrained VGG19 and integrated callback-based training control (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
- Managed large-model artifact workflow (~405 MB checkpoint) and integrated reproducible dependency setup for local deployment.

## Interview Talking Points
- Why transfer learning was chosen over training from scratch.
- Trade-off between model size and deployment simplicity.
- Impact of validation set size on metric stability.
- Steps required to improve clinical reliability.
- How to extend this project for multiclass thoracic disease detection.

## Important Note on Model File and GitHub Limits
The model file is large (`~405 MB`) and may exceed standard GitHub push limits depending on repo setup.
If your GitHub repository does not include this file, keep one of these options:
- Host weights externally and download at runtime.
- Use Git LFS.
- Provide contact/demo link plus reproducible training notebook.

## Reproducibility Checklist
- [ ] Python environment created.
- [ ] Dependencies installed from `requirements.txt`.
- [ ] Correct `vgg19_model_01.h5` available at root.
- [ ] Same input size (`128x128`) enforced.
- [ ] Label mapping preserved (`NORMAL=0`, `PNEUMONIA=1`).

## Author
- Name: Nihal Jaiswal
- GitHub: https://github.com/Nihal108-bi
- LinkedIn: https://www.linkedin.com/in/nihal-jaiswal-908b52257/

## Acknowledgment
- Dataset provider: Kaggle chest X-ray pneumonia dataset by Paul Mooney.

## Final Note
This project is a strong showcase of practical deep learning engineering:
it moves from data ingestion to model training to user-facing deployment,
and provides a solid foundation for production-grade medical AI enhancements.
