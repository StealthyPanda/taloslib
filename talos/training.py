
import torch
import torch.nn.functional as F

from typing import Any
summary_writer = Any
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_exists = True
except ImportError as e:
    print(e)
    print("WARNING: Tensorboard import failed, skipping...")
    tensorboard_exists = False
except AttributeError as e:
    print(e)
    print("WARNING: Tensorboard import failed, skipping...")
    tensorboard_exists = False



import numpy as np

from .models import TalosModule
from . import datapipe
from .utils import gpu_exists, gpu_info, in_notebook
from .logs import logger

from typing import Any, Callable
if not tensorboard_exists:
    class SummaryWriter:
        pass

import time, os


from tqdm import tqdm as tqdm_console
from tqdm.notebook import tqdm as tqdm_notebook

def get_tqdm():
    return tqdm_notebook if in_notebook() else tqdm_console

tqdm = get_tqdm()


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
        board : summary_writer = None, # type: ignore
        metrics : dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        i_mover : Callable = None, o_mover : Callable = None,
        device : str = 'cpu', checkpointing : bool = False
        hooks : dict = None
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
        checkpointing (bool, optional): If `True`, saves model after each epoch.

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
        
        if hooks is not None:
            for hook in hooks['epoch']['before']:
                hook(epoch=ep, model=model)
        
        model.train()
        
        epmetrics = dict()
        
        
        prevtime = time.time()
        starttime = time.time()
        for bi, (inputs, outputs) in enumerate(dataloader):
            if hooks is not None:
                for hook in hooks['batch']['before']:
                    hook(batch=bi, model=model)
            # print(inputs.shape)
            model = model.train()
            
            
            if i_mover: inputs = i_mover(inputs, device=device)
            else: inputs = inputs.to(device)
            
            if o_mover: outputs = o_mover(outputs, device=device)
            else: outputs = outputs.to(device)
            
            ycap = model(inputs)
            
            loss : torch.Tensor = loss_fn(ycap, outputs)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            currtime = time.time()
            eta = (currtime - prevtime) * (nbatches - bi - 1)
            eta = time.strftime("%M:%S", time.gmtime(eta))
            totaleta = (currtime - prevtime) * ((nbatches - bi - 1) + (nbatches * (epochs - ep)))
            totaleta = time.strftime("%M:%S", time.gmtime(totaleta))
            prevtime = currtime
            elapsed = currtime - starttime
            elapsed = time.strftime("%M:%S", time.gmtime(elapsed))
            p = (
                ('=' * int(bi * K / nbatches)) + 
                '>' + (' ' * int((nbatches - bi) * K / nbatches))
            )
            p = p[:K]
            print(
                f'\rEpoch {ep:3} [{p}] loss {loss.item():.6f} ETA [{eta}] Elapsed [{elapsed}] Training ETA [{totaleta}]',
                end=''
            )
            
            if hooks is not None:
                for hook in hooks['batch']['after']:
                    hook(batch=bi, model=model, loss=loss.item())
        
        
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
        
        if checkpointing:
            model.save(f'checkpoints/{model.name}_{ep}', quiet=True)

    return hist




train = generic_train



def empirical_memory_analysis(
        model : TalosModule,
        input_shape : tuple[int, ...], dtype : torch.dtype, sample_target : torch.Tensor,
        forward_fn : Callable[[TalosModule, torch.Tensor], torch.Tensor] = None,
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        batches : list[int] = None, overhead : int = 0,
    ) -> tuple[Callable[[int], int], np.ndarray[int], np.ndarray[int]]:
    
    assert gpu_exists(verbose=False), (
        f'No GPU device found! Cannot perform analysis without GPU as of now...'
    )
    device = 'cuda:0'    
    
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)
    if loss_fn is None:
        loss_fn = lambda ycap, targs: ycap.sum() + targs.sum()
    
    model = model.to(device)
    
    mems = []
    batches = [1, 2, 4, 6] if batches is None else batches
    # for batch sizes 1, 2, 3
    for b in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        ishape = (b, *input_shape)
        tx = torch.zeros(*ishape, dtype=dtype).to(device)
        tycap = forward_fn(model, tx)
        loss = loss_fn(tycap, sample_target)
        loss.backward()
        
        mems.append(torch.cuda.max_memory_allocated(device) - overhead)
    
    # coeffs = np.polyfit(batches, mems, deg=1)
    # predictor = lambda b: coeffs[0]*b + coeffs[1]
    mems = np.array(mems)
    batches = np.array(batches)
    
    print(batches)
    print(mems)
    
    # dely = mems[1:] - mems[:-1]
    # delx = batches[1:] - batches[:-1]
    # slope = (dely / delx).mean().item()
    # predictor = lambda b: slope * b
    coeffs = np.polyfit(batches, mems, deg=2)
    predictor = lambda b: (coeffs[0] * (b**2)) + (coeffs[1]*b) + coeffs[2]
    # predictor = lambda b: (coeffs[0] * (b**2)) + (coeffs[0]*b) + coeffs[1]
    
    return predictor, batches, mems


def get_optimizer_state_size(optimizer) -> tuple[int, int]:
    """Gets the size of optimizer in bytes and no. of elements. Must have its state dict populated 
    (either by `step`ping at least once or `zero_grad`ing at least once).

    Args:
        optimizer: the optimizer.

    Returns:
        tuple[int, int]: (bytes, elements)
    """
    total_bytes = 0
    elems = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total_bytes += value.numel() * value.element_size()
                elems += value.numel()
    return total_bytes, elems



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
            checkpoints : bool = False
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
            checkpoints (bool, optional): If `True`, saves model after each epoch. Defaults to False.
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
        self.checkpoints = checkpoints
        
        self.run = None
        
        self.predictor = None
        
        self.before_batch = []
        self.after_batch = []
        self.before_epoch = []
        self.after_epoch = []
    
    
    def add_hook_before_batch(self, hook : Callable):
        self.before_batch.append(hook)
    def add_hook_after_batch(self, hook : Callable):
        self.after_batch.append(hook)
    def add_hook_before_epoch(self, hook : Callable):
        self.before_epoch.append(hook)
    def add_hook_after_epoch(self, hook : Callable):
        self.after_epoch.append(hook)
    
    
    def analyse_memory(
            self,
            input_shape : tuple[int, ...], dtype : torch.dtype,
            forward_fn : Callable[[TalosModule, torch.Tensor], torch.Tensor] = None,
            batches : list[int] = None,
        ) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """Call this to to a memory requirement analysis on GPU for training.

        Args:
            input_shape (tuple[int, ...]): Shape of input tensors to model.
            forward_fn (Callable[[TalosModule, torch.Tensor], torch.Tensor], optional): A custom way to call the forward in the model. 
                Defaults to simple `model(input_tensor)`.
        """
        sample_targets = self.dataset[0][1]
        self.optim.zero_grad()
                
        model_size = self.model.disk_size()
        optim_size, _ = get_optimizer_state_size(self.optim)
        
        self.predictor, batches, mems = empirical_memory_analysis(
            self.model,
            input_shape, dtype, sample_targets, forward_fn, self.loss_fn, batches,
            overhead = model_size + optim_size
        )
        
        return batches, mems
        
    
    def get_required_memory(self, batch_size : int) -> int:
        """Calculates a rough estimate of GPU memory required as a function of batch_size for this training loop.

        Args:
            batch_size (int): Batch size.

        Returns:
            int: Memory required in bytes.
        """
        assert self.predictor is not None, (
            'Call `analyse_memory` first!'
        )
        
        return self.predictor(batch_size)
    
        
    
    
    def train(self, epochs : int = 1, batch_size : int = None, loss_fn = None):
        
        device = 'cuda' if gpu_exists() else 'cpu'
        
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
        board = None
        if tensorboard_exists:
            board = SummaryWriter(boardname)
            logger.info(f'Logging to {boardname}')
        
        hooks = {
            'epoch' : {
                'before' : self.before_epoch,
                'after' : self.after_epoch,
            },
            'batch' : {
                'before' : self.before_batch,
                'after' : self.after_batch,
            },
        }
        
        hist = generic_train(
            self.model,
            self.dataset, self.testx, self.testy,
            loss_fn, self.optim,
            epochs, batch_size, self.shuffle,
            self.num_workers, self.pin_memory, board,
            self.metrics, self.i_mover, self.o_mover,
            device, self.checkpoints, hooks
        )
        
        self.run += 1
        
        return hist
    
    


