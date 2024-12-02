from torch import nn
import torch


def loss_fn(device):
    return nn.CrossEntropyLoss().to(device)

def optimizer(learning_rate: float, model: nn.Module, optimz: str):
    """
    rmsprop
    
    """
    optim_funcs = {
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
        "sdg": torch.optim.SGD,
    }
    return optim_funcs[optimz](params=model.parameters(), lr=learning_rate)
