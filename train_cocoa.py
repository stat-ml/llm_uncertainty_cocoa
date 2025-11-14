import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
import pickle
import argparse
import random
import os
import re
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt

import wandb
from copy import deepcopy
from models.mlp import MLP
from data.loaders import get_embeddings
from data.utils import load_managers
from typing import List
from lm_polygraph.ue_metrics import PredictionRejectionArea

from lm_polygraph.estimators.greedy_supervised_cocoa import *

from lm_polygraph.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph.estimators.perplexity import Perplexity
from lm_polygraph.estimators.token_entropy import MeanTokenEntropy
from lm_polygraph.estimators.greedy_semantic_average_ue_average_similarity import *

estimators = [SupervisedCocoaMSP(), SupervisedCocoaPPL(), SupervisedCocoaMTE()]
ue_metrics = [PredictionRejectionArea(max_rejection=0.5)]
lower_bound_methods = [MaximumSequenceProbability(),Perplexity(), MeanTokenEntropy()]
upper_bound_methods = [GreedySemanticEnrichedMaxprobAveDissimilarity(), GreedySemanticEnrichedPPLAveDissimilarity(), GreedySemanticEnrichedMTEAveDissimilarity()]


os.environ["WANDB_LOG_MODEL"]="end"

quality_metrics = {
    'triviaqa': 'AlignScoreTargetOutput',
    'coqa':'AlignScoreTargetOutput',
    'gsm8k':'Accuracy',
    'wmt14_fren':'Comet',
    'wmt19_deen':'Comet',
    'mmlu':'Accuracy',
    'xsum':'AlignScoreInputOutput',
}

def set_seed(seed: int = 1):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SaveMetricsCallback(TrainerCallback):
    # def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    #     if metrics:
    #         print("Saving metrics:", metrics) 
    #         with open("metrics.json", "a") as f:
    #             json.dump(metrics, f, indent=4)
    #             f.write("\n")
    #     else:
    #         print("Warning: No metrics were computed!")

    def __init__(self, save_path, dataset_name, pooling_type='mean', layer=-1):
        super().__init__()
        self.save_path = save_path
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = None
        self.dataset_name = dataset_name
        self.pooling_type = pooling_type
        self.layer = layer

    def _extract_dataset_name(self, path):
        """Extracts 'coqa' from paths like '.../llama_coqa_train.man'"""
        filename = os.path.basename(path)
        # Match patterns like llama_coqa_train.man or llama_coqa_test.man
        match = re.search(r'llama_(.*?)_(train|test)\.man', filename)
        return match.group(1) if match else "unknown_dataset"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            self.val_metrics.append(logs.copy())
        elif logs and 'loss' in logs:
            self.train_metrics.append(logs.copy())

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Detect if this is the final test evaluation
        if 'eval_loss' in metrics and state.epoch == args.num_train_epochs:
            self.test_metrics = metrics.copy()
            # Prefix test metrics for clarity
            self.test_metrics = {f"test_{k}": v for k, v in self.test_metrics.items()}

    def on_train_end(self, args, state, control, **kwargs):
        # Save all metrics
        all_metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'test': self.test_metrics
        }
        with open(f"{self.save_path}/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)

        # Plot training and validation losses
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(12, 6))

        train_losses = [m['loss'] for m in self.train_metrics if 'loss' in m]
        val_losses = [m['eval_loss'] for m in self.val_metrics if 'eval_loss' in m]

        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-', label='Training loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation loss')

        if self.test_metrics:
            test_loss = self.test_metrics.get('test_loss')
            plt.axhline(y=test_loss, color='g', linestyle='--', 
                       label=f'Test loss ({test_loss:.3f})')

        plt.title(
            f'{self.dataset_name.upper()} - Training/Validation/Test Loss\n'
            f'(Aggregation: {self.pooling_type}, Layer: {self.layer})'
        )
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f"{self.save_path}/loss_plot.png")
        plt.close()

def train(
    seed,
    device,
    base_model,
    dataset,
    manager_dir,
    save_path,
    selected_layer,
    pooling_type,
    num_train_epochs,
    validation_split,
    load_best_model_at_end,
    greedy_or_sample
):
    if greedy_or_sample == 'greedy':
        if load_best_model_at_end == True:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "cocoa-supervised"),
                name=f"fixed_metrics_{base_model}_{dataset}_{pooling_type}_{selected_layer}",
                tags=[base_model, dataset, pooling_type, str(selected_layer)],
                group="embeddings",
            )
        else:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "cocoa-supervised"),
                name=f"last_model_fixed_metrics_{base_model}_{dataset}_{pooling_type}_{selected_layer}",
                tags=[base_model, dataset, pooling_type, str(selected_layer)],
                group="embeddings",
            )
    elif greedy_or_sample == 'sample':
        if load_best_model_at_end == True:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "cocoa-supervised"),
                name=f"sample_fixed_metrics_{base_model}_{dataset}_{pooling_type}_{selected_layer}",
                tags=[base_model, dataset, pooling_type, str(selected_layer)],
                group="embeddings",
            )
        else:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "cocoa-supervised"),
                name=f"last_model_{base_model}_{dataset}_{pooling_type}_{selected_layer}",
                tags=[base_model, dataset, pooling_type, str(selected_layer)],
                group="embeddings",
            )

    head_dim = 4096
    dropout = 0.1
    interim_dim = 2048

    if base_model == 'falcon':
        hidden_dim_embeds = 3072
    else:
        hidden_dim_embeds = 4096

    set_seed(seed)
    model = MLP(hidden_dim_embeds, head_dim, interim_dim, dropout)
    model.to(device)

    train_manager, test_manager = load_managers(base_model, dataset, manager_dir, device)

    embeddings_train, \
    targets_train, \
    ids_train, \
    embeddings_test, \
    targets_test, \
    ids_test = get_embeddings(train_manager,
                              test_manager,
                              pooling_type,
                              selected_layer, 
                              greedy_or_sample)

    def collate_fn(batch):
        return {
            "embeddings": torch.stack([torch.tensor(sample["embedding"], dtype=torch.float32) for sample in batch], dim=0),
            "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)
        }

    def to_dataset(embeddings, targets):
        dataset = []
        for embedding, target in zip(embeddings, targets):
            dataset.append({
                "embedding": embedding,
                "label": target
            })
        return Dataset.from_list(dataset)

    positions = list(range(len(embeddings_train)))
    train_pos, val_pos = train_test_split(positions, test_size=validation_split, random_state=seed)

    # Split embeddings and targets by position
    train_embeddings = [embeddings_train[i] for i in train_pos]
    train_targets = [targets_train[i] for i in train_pos]

    val_embeddings = [embeddings_train[i] for i in val_pos]
    val_targets = [targets_train[i] for i in val_pos]

    # Map val positions to their original manager indices
    val_manager_ids = [ids_train[i] for i in val_pos]

    val_manager = train_manager

    # Filter stats to keep only the ones that match validation ids
    for stat_key, stat_values in val_manager.stats.items():
        if isinstance(stat_values, list):
            val_manager.stats[stat_key] = [stat_values[i] for i in val_manager_ids]
        elif isinstance(stat_values, np.ndarray):
            val_manager.stats[stat_key] = stat_values[val_manager_ids]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")

    for stat_key, stat_values in val_manager.gen_metrics.items():
        if isinstance(stat_values, list):
            val_manager.gen_metrics[stat_key] = [stat_values[i] for i in val_manager_ids]
        elif isinstance(stat_values, np.ndarray):
            val_manager.gen_metrics[stat_key] = stat_values[val_manager_ids]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")

    train_dataset = to_dataset(train_embeddings, train_targets)
    val_dataset = to_dataset(val_embeddings, val_targets)
    test_dataset = to_dataset(embeddings_test, targets_test)


    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_train_epochs,
        learning_rate=1e-5,
        weight_decay=0.1,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=7, 
        max_grad_norm=1.0,
        logging_strategy="epoch",
        save_strategy="epoch", 
        eval_strategy="epoch",
        fp16=False,
        remove_unused_columns=False,
        report_to="wandb",
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()

        print(f"Predictions: {predictions[:5]}")
        print(f"Labels: {labels[:5]}")
        val_manager.stats['greedy_sentence_similarity_supervised'] = predictions

        quality_metric = quality_metrics[dataset]

        quality_values = val_manager.gen_metrics[('sequence', quality_metric)]
        val_manager.ue_metrics = ue_metrics

        for estimator in estimators:
            values = estimator(val_manager.stats)
            val_manager.estimations[('sequence', str(estimator))] = values

        for estimator in lower_bound_methods:
            values = estimator(val_manager.stats)
            val_manager.estimations[('sequence', str(estimator))] = values

        for estimator in upper_bound_methods:
            values = estimator(val_manager.stats)
            val_manager.estimations[('sequence', str(estimator))] = values

        val_manager.eval_ue()

        
        prr_cocoa_MSP_supervised = val_manager.metrics[('sequence', str(estimators[0]), quality_metric , 'prr_0.5_normalized')]
        prr_MSP = val_manager.metrics[('sequence', str(lower_bound_methods[0]), quality_metric , 'prr_0.5_normalized')]
        prr_cocoa_MSP = val_manager.metrics[('sequence', str(upper_bound_methods[0]), quality_metric , 'prr_0.5_normalized')]

        prr_cocoa_PPL_supervised = val_manager.metrics[('sequence', str(estimators[1]), quality_metric , 'prr_0.5_normalized')]
        prr_PPL = val_manager.metrics[('sequence', str(lower_bound_methods[1]), quality_metric , 'prr_0.5_normalized')]
        prr_cocoa_PPL = val_manager.metrics[('sequence', str(upper_bound_methods[1]), quality_metric , 'prr_0.5_normalized')]

        prr_cocoa_MTE_supervised = val_manager.metrics[('sequence', str(estimators[2]), quality_metric , 'prr_0.5_normalized')]
        prr_MTE = val_manager.metrics[('sequence', str(lower_bound_methods[2]), quality_metric , 'prr_0.5_normalized')]
        prr_cocoa_MTE = val_manager.metrics[('sequence', str(upper_bound_methods[2]), quality_metric , 'prr_0.5_normalized')]


        print("Lengths: " , len(predictions),len(labels))
        
        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        with open(save_path + "predictions.pickle", "wb") as f:
            pickle.dump({"predictions": predictions, "labels": labels}, f)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2   ,  
        # PRR for MSP
        "MSP": prr_MSP,
        "MSP_cocoa": prr_cocoa_MSP,
        "MSP_cocoa_supervised": prr_cocoa_MSP_supervised,
        # PRR for PPL
        "PPL": prr_PPL,
        "PPL_cocoa": prr_cocoa_PPL,
        "PPL_cocoa_supervised": prr_cocoa_PPL_supervised,
        # PRR for MTE
        "MTE": prr_MTE,
        "MTE_cocoa": prr_cocoa_MTE,
        "MTE_cocoa_supervised": prr_cocoa_MTE_supervised,}
        
        print("Computed Metrics:", metrics)
        return metrics

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    metrics_callback = SaveMetricsCallback(save_path, dataset, pooling_type, selected_layer)
    trainer.add_callback(metrics_callback)

    trainer.train()

    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Test set results:", test_results)

    test_preds = trainer.predict(test_dataset)
    with open(os.path.join(save_path, "test_predictions.pickle"), "wb") as f:
        pickle.dump({
            "predictions": test_preds.predictions.flatten(),
            "labels": test_preds.label_ids.flatten()
        }, f)

    wandb.finish()

          
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='coqa'
    )
    parser.add_argument(
        '--manager_dir',
        type=str,
        default='managers'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='workdir/'
    )
    parser.add_argument(
        '--selected_layer',
        type=int,
        default=15,
    )
    parser.add_argument(
        '--pooling_type',
        type=str,
        default='mean'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=2
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--load_best_model_at_end',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--greedy_or_sample',
        type=str,
        default='greedy'
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    train(
        seed=args.seed,
        device=args.device,
        base_model=args.model,
        dataset=args.dataset,
        manager_dir=args.manager_dir,
        save_path=args.save_path,
        selected_layer=args.selected_layer,
        pooling_type=args.pooling_type,
        num_train_epochs=args.num_train_epochs,
        validation_split=args.validation_split,
        load_best_model_at_end=args.load_best_model_at_end,
        greedy_or_sample=args.greedy_or_sample
    )
