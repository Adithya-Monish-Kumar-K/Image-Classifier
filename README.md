# Image Classification (Cats vs Dogs)

A transfer learning project using PyTorch to classify images of cats and dogs (Kaggle Cats and Dogs dataset) with a ResNet18 backbone.

## Project Structure
```
├── data/                # Place raw dataset here (unzipped Kaggle dataset)
├── src/
│   ├── data/
│   │   └── dataset.py   # Dataset and DataLoader utilities
│   ├── models/
│   │   └── model.py     # Model builder (transfer learning)
│   ├── training/
│   │   ├── train.py     # Training loop
│   │   └── evaluate.py  # Evaluation helpers
│   ├── infer.py         # Inference script for single images
│   └── utils.py         # Utility helpers (seed, device, metrics)
├── requirements.txt
└── README.md
```

## Setup
1. Create & activate virtual environment (already configured if using VS Code Python env).
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Download the Kaggle Cats and Dogs dataset (or any cats vs dogs dataset) and extract so you have e.g.:
```
/data/PetImages/Cat/*.jpg
/data/PetImages/Dog/*.jpg
```

## Quick Start (Training)
```
python -m src.training.train --data_dir data/PetImages --epochs 3 --batch_size 32 --img_size 224
```

## Inference
```
python -m src.infer --model_path models/best_model.pt --image path/to/image.jpg
```

## Notes
- Uses pretrained ResNet18 with final layer replaced.
- Freezes early layers by default (can unfreeze with flag).
- Saves best model by validation accuracy.