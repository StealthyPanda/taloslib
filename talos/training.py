
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter

from .models import TalosModule
from . import datapipe
from .utils import gpu_exists
from .logs import logger

from typing import Callable

import time, os




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



def misty(
        model : TalosModule,
        dataset : datapipe.Dataset,
        epochs : int = 1,
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
    l = dataset.nbatches(batch_size=batch_size, time_steps=time_steps) 
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}:')
        sub = []
        bi = 0
        for xb, yb in dataset.get_batch(batch_size=batch_size, time_steps=time_steps):
            print(f'\tBatch {bi + 1}/{l}:')
            if use_gpu:
                xb = xb.to('cuda')
                yb = yb.to('cuda')
            
            for each in range(steps):
                ycap = model(xb)
                
                loss = loss_fn(ycap, yb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'\r\t\tStep {each + 1}/{steps} : {loss.item():.5f}', end = '')
                sub.append(loss.item())
            print()
            bi += 1
        timeline.append(sub)
    
    print(f'Misty has ended training model `{model.name}`!')
    return timeline
    

def generic_train(
        model : TalosModule,
        dataset : torch.utils.data.Dataset, testx = None, testy = None,
        loss_fn = F.mse_loss, optim : torch.optim.Optimizer = None,
        epochs : int = 1, batch_size : int = 32, shuffle : bool = True, num_workers : int = 0,
        pin_memory : bool = True,
        board : SummaryWriter = None,
        metrics : dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        i_mover : Callable = None, o_mover : Callable = None,
        device : str = 'cpu',
    ) -> dict[str, list]:
    """General training algorithm.

    Args:
        model (TalosModule): Model to train.
        dataset (torch.utils.data.Dataset): Training dataset.
        testx (_type_, optional): Test inputs. Defaults to None.
        testy (_type_, optional): Test outputs. Defaults to None.
        loss_fn (_type_, optional): Loss function. Defaults to F.mse_loss.
        optim (torch.optim.Optimizer, optional): Optimizer for the model. Defaults to Adam.
        epochs (int, optional): No. of epochs to train. Defaults to 1.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the dataset each epoch. Defaults to True.
        num_workers (int, optional): Parallel workers for loading batches. Leave this for now. Defaults to 0.
        pin_memory (bool, optional): CUDA optimization. Leave this True for NVIDIA GPUs. Defaults to True.
        board (SummaryWriter, optional): tensorboard for logging. Defaults to None.
        metrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): Additional metrics to track. Defaults to None.
        i_mover (Callable, optional): Function to move inputs to device. Use when non-simple inputs. Defaults to None.
        o_mover (Callable, optional): Function to move outputs to device. Use when non-simple outputs. Defaults to None.
        device (str, optional): Device to train on. Defaults to 'cpu'.

    Returns:
        dict[str, list]: Train and test losses by default, and all additional metrics provided in `metrics`.
    """
    
    
    logger.info(f'Training `{model.name}`')
    
    model = model.to(device)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    nbatches = len(dataloader)
    N = len(dataset)
    K = 30
    
    hist = {
        'train_loss' : [],
        'test_loss' : [],
    }
    
    testexists = (testx is not None) and (testy is not None)
    
    for ep in range(epochs):
        ep += 1
        
        model.train()
        
        epmetrics = dict()
        
        
        prevtime = time.time()
        starttime = time.time()
        for bi, (inputs, outputs) in enumerate(dataloader):
            
            # print(inputs.shape)
            
            if i_mover: inputs = i_mover(inputs)
            else: inputs = inputs.to(device)
            
            if o_mover: outputs = o_mover(outputs)
            else: outputs = outputs.to(device)
            
            ycap = model(inputs)
            
            loss : torch.Tensor = loss_fn(ycap, outputs)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            currtime = time.time()
            eta = (currtime - prevtime) * (nbatches - bi - 1)
            eta = time.strftime("%M:%S", time.gmtime(eta))
            prevtime = currtime
            elapsed = currtime - starttime
            elapsed = time.strftime("%M:%S", time.gmtime(elapsed))
            p = (
                ('=' * int(bi * K / nbatches)) + 
                '>' + (' ' * int((nbatches - bi) * K / nbatches))
            )
            p = p[:K]
            print(
                f'\rEpoch {ep:3} [{p}] loss {loss.item():.6f} ETA [{eta}] Elapsed [{elapsed}]',
                end=''
            )
        
        
        if testexists:
            testx = testx.to(device)
            testy = testy.to(device)
            
            with torch.no_grad():
                
                testycap = model(testx)
                testloss = loss_fn(testycap, testy)
                
                print(
                    f'\rEpoch {ep:3} [{p}] loss {loss.item():.6f} ' + 
                    f'test loss {testloss.item():.6f} ETA [{eta}] Elapsed [{elapsed}]',
                )
                
                model.eval()
                testycap = model(testx)
                if metrics is not None:
                    for each in metrics:
                        epmetrics[f'{each}'] = metrics[each](testycap, testy)
        else: print()
        
        
        if testexists:
            hist['test_loss'].append(testloss.item())
        hist['train_loss'].append(loss.item())
        for each in epmetrics:
            if each in hist:
                hist[each].append(epmetrics[each])
            else:
                hist[each] = [epmetrics[each]]
        
        if board is not None:
            board.add_scalar('Loss/train', loss.item(), ep)
            if testexists:
                board.add_scalar('Loss/test', testloss.item(), ep)
            for each in epmetrics:
                board.add_scalar(f'{each}/test', epmetrics[each], ep)

    return hist




train = generic_train


class Trainer:
    
    def __init__(
            self, 
            model : TalosModule,
            dataset : torch.utils.data.Dataset, testx = None, testy = None,
            loss_fn = F.mse_loss, optim = torch.optim.Adam,
            batch_size : int = 32, shuffle : bool = True, num_workers : int = 0,
            pin_memory : bool = True,
            metrics : dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            i_mover : Callable = None, o_mover : Callable = None,
        ):
        
        """
        Args:
            model (TalosModule): Model to train.
            dataset (torch.utils.data.Dataset): Training dataset.
            testx (_type_, optional): Test inputs. Defaults to None.
            testy (_type_, optional): Test outputs. Defaults to None.
            loss_fn (_type_, optional): Loss function. Defaults to F.mse_loss.
            optim : Optimizer for the model. Defaults to Adam.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the dataset each epoch. Defaults to True.
            num_workers (int, optional): Parallel workers for loading batches. Leave this for now. Defaults to 0.
            pin_memory (bool, optional): CUDA optimization. Leave this True for NVIDIA GPUs. Defaults to True.
            metrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): Additional metrics to track. Defaults to None.
            i_mover (Callable, optional): Function to move inputs to device. Use when non-simple inputs. Defaults to None.
            o_mover (Callable, optional): Function to move outputs to device. Use when non-simple outputs. Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.testx = testx
        self.testy = testy
        self.loss_fn = loss_fn
        self.optim = optim(model.parameters())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.metrics = metrics
        self.i_mover = i_mover
        self.o_mover = o_mover
        self.device = 'cuda' if gpu_exists(False) else 'cpu'
        
        self.run = None
    
    def train(self, epochs : int = 1, batch_size : int = None, loss_fn = None):
        
        if batch_size is None:
            batch_size = self.batch_size
        
        if loss_fn is None:
            loss_fn = self.loss_fn
        
        if self.run is None:
            if os.path.exists('.talos/runs/'):
                n = len(os.listdir('.talos/runs/'))
                self.run = n + 1
            else: self.run = 1
        
        boardname = f'.talos/runs/run_{self.run}'
        board = SummaryWriter(boardname)
        logger.info(f'Logging to {boardname}')
        
        hist = generic_train(
            self.model,
            self.dataset, self.testx, self.testy,
            loss_fn, self.optim,
            epochs, batch_size, self.shuffle,
            self.num_workers, self.pin_memory, board,
            self.metrics, self.i_mover, self.o_mover,
            self.device
        )
        
        self.run += 1
        
        return hist
    
    


