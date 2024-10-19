
import torch
import torch.nn.functional as F

from .models import TalosModule
from . import datapipe

from typing import Callable


def ash_ketchum(
        model : TalosModule,
        dataset : datapipe.Dataset,
        epochs : int,
        batch_size : int = 16,
        time_steps : int = 32,
        steps : int = 100,
        optimizer : torch.optim.Optimizer = None,
        loss_fn : Callable = F.mse_loss,
        use_gpu : bool = True
    ) -> list[list[float]]:
    """Trains the given model on the given dataset.

    Args:
        model (TalosModule): model to train
        dataset (pipe.Dataset): dataset to train on (this must have `.split()` called on it already)
        epochs (int): no. of epochs to train for.
        steps (int, optional): steps to train on each epoch. Defaults to 100.
        optimizer (torch.optim.Optimizer, optional): optimizer object to use. Defaults to Adam, with default values.
        use_gpu (bool, optional): train on GPU. Defaults to True.

    Returns:
        list[list[float]]: training loss over each epoch.
    """
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if use_gpu: model = model.to('cuda')

    timeline = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        
        bx, by = dataset.get_batch(batch_size, time_steps)
        if use_gpu:
            bx = bx.to('cuda')
            by = by.to('cuda')
        
        sub = []
        for each in range(steps):
            ycap = model(bx)
            
            loss = loss_fn(ycap, by)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'\r\tStep {each + 1}/{steps} : {loss.item():.5f}', end = '')
            sub.append(loss.item())
        print()
        timeline.append(sub)

    print('Ash has ended training!')
    return timeline
train = ash_ketchum


def misty(
        model : TalosModule,
        dataset : datapipe.Dataset,
        steps : int = 100,
        batch_size : int = 16,
        time_steps : int = 32,
        optimizer : torch.optim.Optimizer = None,
        loss_fn : Callable = F.mse_loss,
        use_gpu : bool = True
    ) -> list[list[float]]:
    """Trains the given model on the given dataset.

    Args:
        model (TalosModule): model to train
        dataset (pipe.Dataset): dataset to train on (this must have `.split()` called on it already)
        steps (int, optional): steps to train on each epoch. Defaults to 100.
        optimizer (torch.optim.Optimizer, optional): optimizer object to use. Defaults to Adam, with default values.
        use_gpu (bool, optional): train on GPU. Defaults to True.

    Returns:
        list[list[float]]: training loss over each epoch.
    """
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if use_gpu: model = model.to('cuda')

    timeline = []
    epoch = 0
    while True:
        print(f'Epoch {epoch + 1}/{int(dataset.samples / time_steps) + 1}:')
        
        bx, by = dataset.get_batch_new(batch_size=batch_size, time_steps=time_steps)
        if len(bx) == 0: break
        
        if use_gpu:
            bx = bx.to('cuda')
            by = by.to('cuda')
        
        sub = []
        for each in range(steps):
            ycap = model(bx)
            
            loss = loss_fn(ycap, by)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'\r\tStep {each + 1}/{steps} : {loss.item():.5f}', end = '')
            sub.append(loss.item())
        print()
        timeline += (sub)
        epoch += 1
        

    print('Misty has ended training!')
    return timeline



