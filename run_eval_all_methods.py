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

ue_metrics = [PredictionRejectionArea(max_rejection=0.5)]

lower_bound_methods_greedy = [MaximumSequenceProbability(), Perplexity(), MeanTokenEntropy()]
lower_bound_methods_sample = [SampledMaximumSequenceProbability(sample_strategy='best'), SampledPerplexity(sample_strategy='best'), SampledMeanTokenEntropy(sample_strategy='best')]
lower_bound_methods_mbr = [SampledMaximumSequenceProbability(sample_strategy='mbr'), SampledPerplexity(sample_strategy='mbr'), SampledMeanTokenEntropy(sample_strategy='mbr')]


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

upper_bound_methods_mbr =[
      SemanticEnrichedMaxprobAveDissimilarity(sample_strategy='mbr'),
    SemanticEnrichedPPLAveDissimilarity(sample_strategy='mbr'),
    SemanticEnrichedMTEAveDissimilarity(sample_strategy='mbr')  
]

old_model_name ={
    'llama':'llama8b',
    'mistral':'mistral7b',
    'falcon':'falcon7b'
}

old_dataset_name ={
    'triviaqa':'triviaqa',
    'coqa':'coqa',
    'mmlu':'mmlu',
    'gsm8k':'gsm8k',
    'wmt14_fren':'wmt14_fren',
    'wmt19_deen':'wmt19_deen',
    'xsum':'xsum',
}

general_baselines_old= [MonteCarloSequenceEntropy(), MonteCarloNormalizedSequenceEntropy(),  SAR(), DegMat(), EigValLaplacian(), SemanticEntropy()]

consistency = [GreedyAveDissimilarity(), AveDissimilarity(sample_strategy='best'), AveDissimilarity(sample_strategy='mbr'), SupervisedCocoa()]

verbalized = [PTrue(), PTrue(sample_strategy='best'), PTrue(sample_strategy='mbr')]

if __name__ == "__main__":
    models = [ 'llama' ,'mistral', 'falcon']
    datasets =  ['mmlu' , 'triviaqa','coqa', 'gsm8k','wmt14_fren','wmt19_deen','xsum']

    for model in models:
        for dataset in datasets:
            man = UEManager.load(f'./enriched_managers/{model}_{dataset}.man')

            for estimator in verbalized:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values

            for estimator in estimators_greedy:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in lower_bound_methods_greedy:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in lower_bound_methods_sample:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in lower_bound_methods_mbr:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values

            for estimator in upper_bound_methods_greedy:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in upper_bound_methods_sample:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in upper_bound_methods_mbr:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values

            for estimator in general_baselines_old:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values
            
            for estimator in consistency:
                values = estimator(man.stats)
                man.estimations[('sequence', str(estimator))] = values

            man.ue_metrics = ue_metrics
            man.eval_ue()
            man_save_path = 'final_mans'
            man.save_path = os.path.join(man_save_path, f"{model}_{dataset}.man")
            man.save()
