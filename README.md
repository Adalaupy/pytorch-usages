# PyTorch Usages

Work-in-progress playground for building and testing PyTorch models across multiple practical use cases.

## Current Status

Implemented and actively used:

- Core models: `Simple`, `LSTM`, `CNN`
- Shared utility layer in `utils/`
- Use cases:
	- `Stock_Price_Prediction`
	- `Face_Recognition`
- Both script and notebook workflows are present in the repo

## Repository Layout

```text
pytorch-usages/
|- models/                         # model definitions
|- utils/                          # reusable helpers (train/image/text/device/dataset)
|- use_case/
|  |- Stock_Price_Prediction/
|  |  |- train/
|  |  |- predict/
|  |  |- financial_data/
|  |  |- constant/
|  |- Face_Recognition/
|     |- train/
|     |- predict/
|     |- data/
|- train_sample/                   # training notebooks
|- predict_sample/                 # prediction notebooks
|- requirements.txt
|- pyproject.toml
```

## Models

| Model | File | Typical use |
|---|---|---|
| Simple feedforward | `models/simple_model.py` | baseline tabular/text tasks |
| LSTM | `models/lstm_model.py` | sequence modeling |
| CNN | `models/cnn_model.py` | image classification |

## Utilities

`utils/` currently includes:

- `train_helper.py`: training loop, early stopping, evaluation helpers
- `image_helper.py`: transforms, face detection, image statistics
- `dataset_helper.py`: Kaggle/HuggingFace download helpers, image dataset utilities
- `text_helper.py`: NLP cleanup and encoding helpers
- `device.py`: checkpoint loading and device selection
- `support_helper.py`: stock-related date and plotting helpers

## Environment Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Python version: `>=3.10` (from `pyproject.toml`).

## Run Examples

### 1) Stock Price Prediction

Train:

```bash
python use_case/Stock_Price_Prediction/train/price_predict_train.py
```

Predict:

```bash
python use_case/Stock_Price_Prediction/predict/price_predict.py
```

Hyperparameter experiment:

```bash
python use_case/Stock_Price_Prediction/train/hyperparameter_experiments.py
```

### 2) News Sentiment (Stock Use Case)

Sentiment analysis is one important element for better stock price prediction because it reflects market reaction to news and events.

Train:

```bash
python use_case/Stock_Price_Prediction/train/news_sentiment_train.py
```

Predict:

```bash
python use_case/Stock_Price_Prediction/predict/news_sentiment_predict.py
```

### 3) Face Recognition

Train:

```bash
python "use_case/Face_Recognition/train/face recognition_train.py"
```

Predict:

```bash
python use_case/Face_Recognition/predict/face_recognition_predict.py
```

## Notes

- Some workflows support Kaggle source download via `kagglehub`.
- Face recognition utilities support optional face detection preprocessing.
- Sample notebooks are available under `train_sample/` and `predict_sample/`.
