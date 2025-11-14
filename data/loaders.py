import torch
import numpy as np


def embeddings_key(pooling_type, greedy_or_sample):
    """
    Return the key for the embeddings in the stats dict according to the pooling type.
    
    Args:
        pooling_type (str): Type of pooling ('mean' or 'last')
    
    Returns:
        str: Key for accessing the pooled embeddings
    """
    if greedy_or_sample == 'greedy':
        return f"{pooling_type}_all_layers_embeddings"
    elif greedy_or_sample == 'sample':
        return f"best_{pooling_type}_all_layers_embeddings"

def pool_embeddings(manager, pooling_type, greedy_or_sample):
    """
    Perform pooling of embeddings over generated tokens.
    
    Args:
        manager (dict): Manager containing embeddings and stats
        pooling_type (str): Type of pooling to perform ('mean' or 'last')
    
    Returns:
        dict: Manager with added pooled embeddings
    """
    pooled_key = embeddings_key(pooling_type, greedy_or_sample)

    # Only compute pooled embeddings if not already cached
    if pooled_key not in manager.stats:
        all_layers_embeddings = manager.stats['all_layers_embeddings']

        pooled_embeddings = []
        for i, sample in enumerate(all_layers_embeddings): 
            pooled_sample = {}
            for layer in range(len(sample)):
                key = f"layer_{layer}_embeddings"
                if pooling_type == 'mean':
                    # Average embeddings across tokens
                    pooled_sample[key] = torch.tensor(sample[key], dtype=torch.float32).mean(dim=0)
                elif pooling_type == 'last':
                    # Take embedding of last token
                    pooled_sample[key] = torch.tensor(sample[key], dtype=torch.float32)[-1]
            pooled_embeddings.append(pooled_sample)

        # Cache the pooled embeddings
        manager.stats[pooled_key] = pooled_embeddings

    return manager

def get_embeddings(
    train_manager,
    test_manager,
    pooling_type,
    layer=-1,
    greedy_or_sample='greedy'
):
    """
    Extract embeddings and targets from managers for training.
    
    Args:
        train_manager (dict): Manager containing training data
        test_manager (dict): Manager containing test data
        pooling_type (str): Type of pooling to use ('mean' or 'last')
        layer (int): Which layer's embeddings to extract (-1 for last layer)
    
    Returns:
        tuple: (train_embeddings, train_targets, train_ids,
                test_embeddings, test_targets, test_ids)
    """
    # Pool embeddings for both train and test sets
    train_manager = pool_embeddings(train_manager, pooling_type, greedy_or_sample)
    test_manager = pool_embeddings(test_manager, pooling_type, greedy_or_sample)

    # Get similarity scores (targets)

    if greedy_or_sample == 'greedy':
        target_key = 'greedy_sentence_similarity'
        train_scores, test_scores = train_manager.stats[target_key], test_manager.stats[target_key]
    elif greedy_or_sample == 'sample':
        # target_key = 'sample_sentence_similarity'
        train_scores = []
        for i in range(len(train_manager.stats['sample_sentence_similarity'])):
            idx = train_manager.stats['best_sample_text_ids'][i]
            row = train_manager.stats['sample_sentence_similarity'][i][idx]
            row = np.delete(row, idx)
            train_scores.append(row)

        test_scores = []
        for i in range(len(test_manager.stats['sample_sentence_similarity'])):
            idx = test_manager.stats['best_sample_text_ids'][i]
            row = test_manager.stats['sample_sentence_similarity'][i][idx]
            row = np.delete(row, idx)
            test_scores.append(row)

    # Get pooled embeddings
    pooled_key = embeddings_key(pooling_type, greedy_or_sample)
    pooled_train_embeddings = train_manager.stats[pooled_key]
    pooled_test_embeddings = test_manager.stats[pooled_key]

    layer_key = f"layer_{layer}_embeddings"

    # For some reason (probably due to empty generation), some samples may not have embeddings for all layers
    # Thus we need to keep track of the ids of the samples that have embeddings for the selected layer
    train_embeddings = []
    train_targets = []
    train_ids = []
    for i, (embs, targs) in enumerate(zip(pooled_train_embeddings, train_scores)):
        if layer_key in embs:
            train_ids.append(i)
            train_embeddings.append(embs[layer_key])
            train_targets.append(1-targs.mean(-1))  # Average similarity scores

    test_embeddings = []
    test_targets = []
    test_ids = []
    for i, (embs, targs) in enumerate(zip(pooled_test_embeddings, test_scores)):
        if layer_key in embs:
            test_ids.append(i)
            test_embeddings.append(embs[layer_key])
            test_targets.append(1-targs.mean(-1))  # Average similarity scores

    return train_embeddings, train_targets, train_ids, test_embeddings, test_targets, test_ids



def get_embeddings_test(
    test_manager,
    pooling_type,
    layer=-1,
    greedy_or_sample='greedy'
):
    """
    Extract embeddings and targets from managers for training.
    
    Args:
        train_manager (dict): Manager containing training data
        test_manager (dict): Manager containing test data
        pooling_type (str): Type of pooling to use ('mean' or 'last')
        layer (int): Which layer's embeddings to extract (-1 for last layer)
    
    Returns:
        tuple: (train_embeddings, train_targets, train_ids,
                test_embeddings, test_targets, test_ids)
    """
    # Pool embeddings for both train and test sets
    test_manager = pool_embeddings(test_manager, pooling_type,greedy_or_sample=greedy_or_sample)

    # Get similarity scores (targets)
    # target_key = 'greedy_sentence_similarity'
    # test_scores= test_manager.stats[target_key]
    # Get pooled embeddings
    # pooled_key = embeddings_key(pooling_type)
    # pooled_test_embeddings = test_manager.stats[pooled_key]

    # layer_key = f"layer_{layer}_embeddings"

    if greedy_or_sample == 'greedy':
        target_key = 'greedy_sentence_similarity'
        test_scores =  test_manager.stats[target_key]
    elif greedy_or_sample == 'sample':
        # target_key = 'sample_sentence_similarity'

        test_scores = []
        for i in range(len(test_manager.stats['sample_sentence_similarity'])):
            idx = test_manager.stats['best_sample_text_ids'][i]
            row = test_manager.stats['sample_sentence_similarity'][i][idx]
            row = np.delete(row, idx)
            test_scores.append(row)

    # Get pooled embeddings
    pooled_key = embeddings_key(pooling_type, greedy_or_sample)
    # pooled_train_embeddings = train_manager.stats[pooled_key]
    pooled_test_embeddings = test_manager.stats[pooled_key]

    layer_key = f"layer_{layer}_embeddings"


    # For some reason (probably due to empty generation), some samples may not have embeddings for all layers
    # Thus we need to keep track of the ids of the samples that have embeddings for the selected layer
 
    test_embeddings = []
    test_targets = []
    test_ids = []
    for i, (embs, targs) in enumerate(zip(pooled_test_embeddings, test_scores)):
        if layer_key in embs:
            test_ids.append(i)
            test_embeddings.append(embs[layer_key])
            test_targets.append(1-targs.mean(-1))  # Average similarity scores

    return test_embeddings, test_targets, test_ids
