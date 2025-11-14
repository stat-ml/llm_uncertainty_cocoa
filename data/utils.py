import os
import torch
from lm_polygraph.utils import UEManager

def load_managers(model, dataset, base_dir, device):
    """
    Load train and test manager files containing model embeddings and statistics.

    Args:
        model (str): Name of the model (e.g., 'llama')
        dataset (str): Name of the dataset (e.g., 'coqa')
        base_dir (str): Base directory containing manager files
        device (str): Device to load the tensors to ('cpu' or 'cuda')

    Returns:
        tuple: (manager_train, manager_test) containing the loaded manager data
    """
    if model=='gemma':
        man_path = os.path.join(base_dir, f'{model}_{dataset}_{{}}.man')
    else:
        man_path = os.path.join(base_dir, f'{model}_{dataset}_{{}}_sample.man' )

    manager_train = UEManager.load(
        man_path.format('train')
        # weights_only=False,
        # map_location=torch.device(device)
    )

    manager_test = UEManager.load(
        man_path.format('test')
        # weights_only=False,
        # map_location=torch.device(device)
    )

    return manager_train, manager_test



def load_test_manager(model, dataset, base_dir, device):
    """
    Load train and test manager files containing model embeddings and statistics.

    Args:
        model (str): Name of the model (e.g., 'llama')
        dataset (str): Name of the dataset (e.g., 'coqa')
        base_dir (str): Base directory containing manager files
        device (str): Device to load the tensors to ('cpu' or 'cuda')

    Returns:
        tuple: (manager_train, manager_test) containing the loaded manager data
    """
    if model=='gemma':
        man_path = os.path.join(base_dir, f'{model}_{dataset}_{{}}.man')
    else:
        man_path = os.path.join(base_dir, f'{model}_{dataset}_{{}}_sample.man' )

    manager_test = UEManager.load(
        man_path.format('test')
    )

    return  manager_test
