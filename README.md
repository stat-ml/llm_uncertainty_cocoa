# Supervised CoCoA Training

This repository contains code to train a small model to predict similarity scores based on saved model features (embeddings).

## Requirements
- PyTorch
- Transformers
- NumPy
- scikit-learn
- wandb (for experiment tracking)

## Setup

1. Create a new virtual environment:
```bash
conda create -n supervised_cocoa python=3.10
conda activate supervised_cocoa
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
export WANDB_PROJECT=cocoa-supervised  # Required for logging
```

## Training

Basic usage:
```bash
python train_cocoa.py \
    --dataset coqa \
    --manager_dir "/path/to/managers" \
    --save_path "workdir/" \
    --selected_layer 15 \
    --pooling_type mean \
    --num_train_epochs 20
```

For grid search across layers:
```bash
bash grid_search_layers_mean.sh
```

## Key Parameters

- `--dataset`: Dataset name (e.g., coqa, trivia)
- `--selected_layer`: Which transformer layer to use (-1 for last layer)
- `--pooling_type`: How to aggregate token embeddings (mean or last)
- `--num_train_epochs`: Number of training epochs
- `--validation_split`: Fraction of training data to use for validation

## Project Structure

- `models/mlp.py`: MLP model architecture
- `data/loaders.py`: Data loading and preprocessing utilities
- `data/utils.py`: Helper functions for loading managers
- `train_cocoa.py`: Main training script
