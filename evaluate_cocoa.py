import wandb
import torch
import numpy as np
import pickle
import json
import argparse
import os
from transformers import Trainer
from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from safetensors.torch import load_file
from lm_polygraph.utils.manager import UEManager

from models.mlp import MLP
from data.loaders import get_embeddings, get_embeddings_test
from data.utils import load_managers, load_test_manager
from lm_polygraph.ue_metrics import PredictionRejectionArea

from lm_polygraph.estimators.greedy_supervised_cocoa import *
from lm_polygraph.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph.estimators.perplexity import Perplexity
from lm_polygraph.estimators.token_entropy import MeanTokenEntropy
from lm_polygraph.estimators.greedy_semantic_average_ue_average_similarity import *
from lm_polygraph.estimators import *


quality_metrics = {
    'triviaqa': 'AlignScoreTargetOutput',
    'coqa': 'AlignScoreTargetOutput',
    'gsm8k': 'Accuracy',
    'wmt14_fren': 'Comet',
    'wmt19_deen': 'Comet',
    'mmlu': 'Accuracy',
    'xsum': 'AlignScoreInputOutput',
}

estimators_greedy = [SupervisedCocoaMSP(), SupervisedCocoaPPL(), SupervisedCocoaMTE()]
estimators_sample = [SampledSupervisedCocoaMSP(sample_strategy='best'), SampledSupervisedCocoaPPL(sample_strategy='best'), SampledSupervisedCocoaMTE(sample_strategy='best')]


ue_metrics = [PredictionRejectionArea(max_rejection=0.5)]

lower_bound_methods_greedy = [MaximumSequenceProbability(), Perplexity(), MeanTokenEntropy()]
lower_bound_methods_sample = [SampledMaximumSequenceProbability(sample_strategy='best'), SampledPerplexity(sample_strategy='best'), SampledMeanTokenEntropy(sample_strategy='best')]


upper_bound_methods_greedy = [
    GreedySemanticEnrichedMaxprobAveDissimilarity(),
    GreedySemanticEnrichedPPLAveDissimilarity(),
    GreedySemanticEnrichedMTEAveDissimilarity()
]
upper_bound_methods_sample =[
      SemanticEnrichedMaxprobAveDissimilarity(sample_strategy='best'),
    SemanticEnrichedPPLAveDissimilarity(sample_strategy='best'),
    SemanticEnrichedMTEAveDissimilarity(sample_strategy='best')  
]

old_model_name ={
    'llama':'llama8b',
    'mistral':'mistral7b',
    'falcon':'falcon7b'
}

old_dataset_name ={
    'triviaqa':'trivia',
    'coqa':'coqa_no_context',
    'mmlu':'mmlu',
    'gsm8k':'gsm8k_cot',
    'wmt14_fren':'wmt14_fren',
    'wmt19_deen':'wmt19_deen',
    'xsum':'xsum',
}

general_baselines_old= [MonteCarloSequenceEntropy(), MonteCarloNormalizedSequenceEntropy(),  SAR()]



def collate_fn(batch):
    return {
        "embeddings": torch.stack([torch.tensor(sample["embedding"], dtype=torch.float32) for sample in batch], dim=0),
        "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)
    }

def to_dataset(embeddings, targets):
    return Dataset.from_list([{"embedding": emb, "label": label} for emb, label in zip(embeddings, targets)])


def evaluate_model(model_path, base_model, dataset, manager_dir, pooling_type, selected_layer, device, man_save_path='.', greedy_or_sample='greedy'):
    head_dim = 4096
    dropout = 0.1
    interim_dim = 2048

    if base_model == 'falcon':
        hidden_dim_embeds = 3072
    elif base_model=='gemma':
        hidden_dim_embeds=3840
    else:
        hidden_dim_embeds = 4096

    # set_seed(seed)
    model = MLP(hidden_dim_embeds, head_dim, interim_dim, dropout)
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if greedy_or_sample=='greedy':
        general_baselines = general_baselines_old+[SupervisedCocoa(), GreedyAveDissimilarity()]
    else:
        general_baselines = general_baselines_old + [SampledSupervisedCocoa(), AveDissimilarity(sample_strategy='best')]
    
    quality_metric = quality_metrics[dataset]

    test_manager = load_test_manager(base_model, dataset, manager_dir, device)
    embeddings_test, targets_test, ids_test = get_embeddings_test( test_manager, pooling_type, selected_layer, greedy_or_sample)
    print("Original length: ", len(test_manager.gen_metrics[('sequence', quality_metric)]))
    print("After filtering: " , len(ids_test))

    old_manager = test_manager
    test_dataset = to_dataset(embeddings_test, targets_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # Filter test manager in case empy embeddings present
    print("Working on new manager")
    for stat_key, stat_values in test_manager.stats.items():
        if isinstance(stat_values, list):
            test_manager.stats[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            test_manager.stats[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")

    for stat_key, stat_values in test_manager.gen_metrics.items():
        if isinstance(stat_values, list):
            test_manager.gen_metrics[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            test_manager.gen_metrics[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")

    for stat_key, stat_values in test_manager.estimations.items():
        if isinstance(stat_values, list):
            test_manager.estimations[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            test_manager.estimations[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")
    print("Working on old manager")
    for stat_key, stat_values in old_manager.stats.items():
        if isinstance(stat_values, list):
            old_manager.stats[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            old_manager.stats[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")
    print('Filtering gen metrics')
    for stat_key, stat_values in old_manager.gen_metrics.items():
        if isinstance(stat_values, list):
            old_manager.gen_metrics[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            old_manager.gen_metrics[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")
    print('Filtering estimations')
    for stat_key, stat_values in old_manager.estimations.items():
        if isinstance(stat_values, list):
            old_manager.estimations[stat_key] = [stat_values[i] for i in ids_test]
        elif isinstance(stat_values, np.ndarray):
            old_manager.estimations[stat_key] = stat_values[ids_test]
        else:
            print(f"Unknown type in stats: {stat_key} — skipping")

    print('Loading embeddings')
    all_preds, all_labels = [], []
    for batch in test_loader:
        inputs = batch["embeddings"].to(device)
        with torch.no_grad():
            preds = model(inputs)
            if isinstance(preds, dict):  
                preds = preds["logits"]  
            preds = preds.squeeze().cpu().numpy()
        labels = batch["labels"].numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    print('Predict')
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    print('Set stats')
    if greedy_or_sample=='greedy':
        old_manager.stats["greedy_sentence_similarity_supervised"] = all_preds
    else:
        old_manager.stats["supervised_sample_sentence_similarity"] = all_preds
    
    old_manager.ue_metrics = ue_metrics

    if greedy_or_sample=='greedy':
        estimators = estimators_greedy
    else:
        estimators = estimators_sample

    print('Being eval')
    for estimator in estimators:
        print(str(estimator))
        values = estimator(old_manager.stats)
        old_manager.estimations[('sequence', str(estimator))] = values

    if greedy_or_sample=='greedy':
        lower_bound_methods = lower_bound_methods_greedy
    else:
        lower_bound_methods = lower_bound_methods_sample

    for estimator in lower_bound_methods:
        print(str(estimator))
        values = estimator(old_manager.stats)
        old_manager.estimations[('sequence', str(estimator))] = values

    if greedy_or_sample=='greedy':
        upper_bound_methods = upper_bound_methods_greedy
    else:
        upper_bound_methods = upper_bound_methods_sample

    for estimator in upper_bound_methods:
        print(str(estimator))
        values = estimator(old_manager.stats)
        old_manager.estimations[('sequence', str(estimator))] = values
    
    for estimator in general_baselines:
        print(str(estimator))
        values = estimator(old_manager.stats)
        old_manager.estimations[('sequence', str(estimator))] = values

    print("Eval ue")
    old_manager.eval_ue()
    for key in list(old_manager.stats.keys()):
        if "embedding" in key:
            old_manager.stats[key] = []
    old_manager.save_path = os.path.join(man_save_path, f"{base_model}_{dataset}.man")
    old_manager.save()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset", type=str, default="triviaqa")
    parser.add_argument("--manager_dir", type=str, default="YOUR_PATH")
    parser.add_argument("--selected_layer", type=int, default=16)
    parser.add_argument("--pooling_type", type=str, default="mean")
    parser.add_argument("--save_dir", type=str, default="enriched_managers")
    parser.add_argument("--run_name", type=str, default="", help="Name of the W&B run")
    parser.add_argument("--greedy_or_sample", type=str, default="greedy")
    return parser.parse_args()


def get_run_id_from_name(project_path: str, run_name: str):
    api = wandb.Api()
    runs = api.runs(project_path)
    for run in runs:
        if run.name == run_name:
            return run.id  # This is what you need for artifact name
    raise ValueError(f"Run name '{run_name}' not found in project '{project_path}'")

if __name__ == "__main__":
    args = parse_args()
    models = ['llama','mistral', 'falcon']
    datasets =  ['gsm8k','wmt14_fren','wmt19_deen', 'xsum','mmlu', 'coqa','triviaqa']

    wandb.login()
    wandb.init()
    project_path = "YOUR_PROJECT_PATH" 

    for model in models:
        for dataset in datasets:
            print(f"Working on {dataset} and {model}")
            try:
                args.model=model
                args.dataset=dataset
                if model=='falcon':
                    args.selected_layer=14
                elif model=='gemma':
                    args.selected_layer = 21
                else:
                    args.selected_layer=16
                
                args.run_name = f"last_model_fixed_metrics_{model}_{dataset}_mean_{str(args.selected_layer)}_again"
                
                run_id = get_run_id_from_name(project_path, args.run_name)

                artifact_name = f"model-{run_id}"
                print('Loading artifact')
                artifact = wandb.use_artifact(f"{project_path}/{artifact_name}:latest", type="model")

                local_path = os.path.join("artifacts", f"{artifact_name}:v1")

                if not os.path.exists(local_path):
                    model_path = artifact.download()
                else:
                    model_path = local_path
                print(f'Model downloaded to {model_path}')
                os.makedirs(args.save_dir, exist_ok=True)

                evaluate_model(
                    model_path=model_path,
                    base_model=args.model,
                    dataset=args.dataset,
                    manager_dir=args.manager_dir,
                    pooling_type=args.pooling_type,
                    selected_layer=args.selected_layer,
                    device=args.device,
                    man_save_path=args.save_dir,
                    greedy_or_sample='greedy'
                )
            except Exception as ex:
                print(f"No model for dataset {dataset}, model {model}, {ex}" )
                print(ex)

