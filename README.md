# CoCoA: A Minimum Bayes Risk Framework Bridging Confidence and Consistency for Uncertainty Quantification in LLMs. 

This repository contains the code required to reproduce the experiments from the paper:
"CoCoA: A Minimum Bayes Risk Framework Bridging Confidence and Consistency for Uncertainty Quantification in LLMs". CoCoA introduces a principled approach for evaluating and improving the calibration of large language models by integrating confidence and consistency under a unified Minimum Bayes Risk (MBR) formulation. 

## Overview

This repository includes:
- Scripts to **run inference** using extended [LM-Polygraph](https://github.com/silvimica/lm-polygraph/tree/cocoa_supervised).
- Tools to **train CoCoA light** variations.
- Utilities for **extracting results** and **building tables**.

## Project structure

```
├── models/
│   └── mlp.py              # MLP model architecture
├── data/
│   ├── loaders.py          # Data loading and preprocessing
│   └── utils.py            # Helper functions for loading managers
├── results_csv/            # Recorded results
├── train_cocoa.py          # Main training script
├── evaluate_cocoa.py       # Evaluation script
├── run_eval_all_methods.py # Metric computation
└── extract_results.py      # Aggregate and export results
```

## Running LM-Polygraph

Inference relies on a compatible version of **LM-Polygraph**.

👉 Link to the specific LM-Polygraph repository version will be added here:  
`[LM-Polygraph Repository (specific CoCoA version)](https://github.com/silvimica/lm-polygraph/tree/cocoa_supervised)`

### Setup

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

### Running inference

Within lm-polygraph directory, use the followgin command to run inference:

```bash
HF_HOME=/path/to/cache HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_coqa_cocoa_supervised.yaml polygraph_eval cache_path=/path/to/cache eval_split='validation' subsample_eval_dataset=2000 model=llama batch_size=1
```

## CoCoA Light Training

### Training

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

#### Key Parameters

- `--dataset`: Dataset name (e.g., coqa, trivia)
- `--selected_layer`: Which transformer layer to use (-1 for last layer)
- `--pooling_type`: How to aggregate token embeddings (mean or last)
- `--num_train_epochs`: Number of training epochs
- `--validation_split`: Fraction of training data to use for validation

#### Project Structure

- `models/mlp.py`: MLP model architecture
- `data/loaders.py`: Data loading and preprocessing utilities
- `data/utils.py`: Helper functions for loading managers
- `train_cocoa.py`: Main training script

### Evaluating

To enrich experimental managers with a predicted Consistency scores, run the following script:

```bash
python evaluate_cocoa.py
```

## Extracting Results

To evaluate all computed metrics based on collected statistics, use the following command: 

```bash
python run_eval_all_methods.py
```


After inference and training are complete, you can extract and record results to csv using:

```bash
python extract_results.py 
```

In *build_tables.ipynb* you can find scipts to produce the tables from main parts of the paper and detailed experimantal results. 

## Citation

If you find this work useful, please cite:

```
@inproceedings{
vashurin2025cocoa,
title={CoCoA: A Minimum Bayes Risk Framework Bridging Confidence and Consistency for Uncertainty Quantification in {LLM}s},
author={Roman Vashurin and Maiya Goloburda and Albina Ilina and Aleksandr Rubashevskii and Preslav Nakov and Artem Shelmanov and Maxim Panov},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=H1NGlLNaVC}
}
```


