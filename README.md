# PyTorch Usages

> **Work in Progress** — This project is still actively being developed. Content, models, and use cases are subject to change.

## Purpose

This project aims to build a foundation for experimenting with different neural network models using PyTorch. It serves as a learning and reference base for model building, training, and applying models to real-world use cases.

## Project Structure

```
pytorch-usages/
├── models/          # Core model definitions
├── utils/           # Shared utility functions
├── train_sample/    # Training notebooks
├── predict_sample/  # Prediction notebooks
├── use_case/        # Applied use cases
└── data/            # Datasets
```

## Models

Located in the `models/` folder. The following model architectures are available:

| Model | File | Description |
|---|---|---|
| Simple (Feedforward) | `simple_model.py` | Basic fully connected neural network |
| LSTM | `lstm_model.py` | Long Short-Term Memory network for sequential data |
| CNN | `cnn_model.py` | Convolutional Neural Network for image data |

## Utilities

Located in the `utils/` folder. Common helper functions shared across models and use cases:

- `device.py` — Device setup (CPU / GPU selection)
- `image_helper.py` — Image loading and preprocessing
- `text_helper.py` — Text processing helpers
- `train_helper.py` — Training loop and checkpoint utilities
- `support_helper.py` — General support functions

## Use Cases

Located in the `use_case/` folder. Each use case applies one or more models to achieve a specific goal:

| Use Case | Description |
|---|---|
| Stock Price Prediction | Predicts stock prices using financial data |

## Requirements

- Python 3.x
- PyTorch
- See `requirements.txt` for the full list of dependencies

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script
python main.py
```

## After editing pyproject.toml

If you update or create `pyproject.toml`, install your project in editable mode:

```bash
pip install -e .
```

Or, if you use Poetry:

```bash
poetry install
```

## Hyperparameter Search

To automatically search for the best combination of parameters, you can run:

```bash
python use_case/Stock_Price_Prediction/train/hyperparameter_experiments.py
```
This will execute a grid search and print the best results found.
```
