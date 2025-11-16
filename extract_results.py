from lm_polygraph.utils import UEManager

import pandas as pd 

def get_metrics(decoding = 'greedy'):
    if decoding=='greedy':
        return  {
                    'triviaqa': 'AlignScoreOutputTarget',
                    'coqa': 'AlignScoreOutputTarget',
                    'gsm8k': 'Accuracy',
                    'wmt14_fren': 'Comet',
                    'wmt19_deen': 'Comet',
                    'mmlu': 'Accuracy',
                    'xsum': 'AlignScoreInputOutput',
                }
    elif decoding=='sample':
        return {
                    'triviaqa': 'BestSampleAlignScoreOutputTarget',
                    'coqa': 'BestSampleAlignScoreOutputTarget',
                    'gsm8k': 'BestSampleAccuracy',
                    'wmt14_fren': 'BestSampleComet',
                    'wmt19_deen': 'BestSampleComet',
                    'mmlu': 'BestSampleAccuracy',
                    'xsum': 'BestSampleAlignScoreInputOutput',
                }
    elif decoding=='mbr':
        return {
            'triviaqa': 'MbrSampleAlignScoreOutputTarget',
            'coqa': 'MbrSampleAlignScoreOutputTarget',
            'gsm8k': 'MbrSampleAccuracy',
            'wmt14_fren': 'MbrSampleComet',
            'wmt19_deen': 'MbrSampleComet',
            'mmlu': 'MbrSampleAccuracy',
            'xsum': 'MbrSampleAlignScoreInputOutput',
        }
    else:
        raise ValueError(f"Unknown decoding mode: {decoding}")




def get_methods(decoding = 'greedy'):
    if decoding=='greedy':
        methods =[
                'MonteCarloSequenceEntropy',
                'MonteCarloNormalizedSequenceEntropy',
                'SemanticEntropy',
                'DegMat_NLI_score_entail',
                'EigValLaplacian_NLI_score_entail',
                'SAR_t0.001' ,
                'PTrue',
                'GreedyAveDissimilarity',
                'SupervisedCocoa',
                'MaximumSequenceProbability',
                'GreedySemanticEnrichedMaxprobAveDissimilarity',
                'SupervisedCocoaMSP',
                'Perplexity',
                'GreedySemanticEnrichedPPLAveDissimilarity',
                'SupervisedCocoaPPL',
                'MeanTokenEntropy',
                'GreedySemanticEnrichedMTEAveDissimilarity',
                'SupervisedCocoaMTE'
            ]
    elif decoding =='sample':
        methods = [
                'MonteCarloSequenceEntropy',
                'MonteCarloNormalizedSequenceEntropy',
                'SemanticEntropy',
                'DegMat_NLI_score_entail',
                'EigValLaplacian_NLI_score_entail',
                'SAR_t0.001',
                'PTrueBestSample',
                'BestAveDissimilarity',
                'BestSampledMaximumSequenceProbability',
                'BestSemanticEnrichedMaxprobAveDissimilarity',
                'BestSampledPerplexity',
                'BestSemanticEnrichedPPLAveDissimilarity',
                'BestSampledMeanTokenEntropy',
                'BestSemanticEnrichedMTEAveDissimilarity',
            ]
    elif decoding =='mbr':
        methods = [
                    'MonteCarloSequenceEntropy',
                    'MonteCarloNormalizedSequenceEntropy',
                    'SemanticEntropy',
                    'DegMat_NLI_score_entail',
                    'EigValLaplacian_NLI_score_entail',
                    'SAR_t0.001',
                    'PTrueMbrSample',
                    'MbrAveDissimilarity',
                    'MbrSampledMaximumSequenceProbability',
                    'MbrSemanticEnrichedMaxprobAveDissimilarity',
                    'MbrSampledPerplexity',
                    'MbrSemanticEnrichedPPLAveDissimilarity',
                    'MbrSampledMeanTokenEntropy',
                    'MbrSemanticEnrichedMTEAveDissimilarity',
                ]
    else:
        raise ValueError(f"Unknown decoding mode: {decoding}")

    return methods


def get_results(models, datasets, decoding):
    methods_dict = get_methods(decoding=decoding)
    metrics = get_metrics(decoding=decoding)

    results = []
    for model in models:
        for dataset in datasets:
            print(f'Processing {dataset}, {model}')
            man = UEManager.load(f'./final_mans/{model}_{dataset}.man')
            for method in methods_dict:
                prr_score = man.metrics[('sequence', method, metrics[dataset], 'prr_0.5_normalized')]
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'method': method,
                    'score': prr_score
                })
    df = pd.DataFrame(results)
    df.to_csv(f'results_csv/{decoding}_full_results.csv')

def main():
    models = ['llama', 'mistral', 'falcon']
    datasets = ['triviaqa','coqa','mmlu','gsm8k','xsum','wmt14_fren','wmt19_deen']
    decodings = [ 'greedy' , 'sample' ,'mbr']
    for decoding in decodings:
        get_results(decoding=decoding, datasets=datasets, models=models)


    
if __name__ == '__main__':
    main()

