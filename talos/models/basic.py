
import os

import torch.nn as nn

from ..utils import talosdir, Tensor, tensor
from safetensors.torch import save_model, load_model


modelsavesdir = os.path.join(talosdir, 'models')


class TalosModule(nn.Module):
    """Better module than a simple nn.Module (which it subclasses), adding more functionality."""
    
    n = 0
    def __init__(self, name : str = None, *args, **kwargs) -> None:
        self.name = name
        if name is None:
            self.name = f'module_{TalosModule.n}'
            TalosModule.n += 1
        super().__init__(*args, **kwargs)
    
    def nparams(self) -> int:
        """Returns all the params in this model."""
        n = sum(list(map(lambda x:x.numel(), self.parameters(recurse=True))))
        return n

    def freeze(self, immediate_only : bool = False) -> None:
        """Freezes the entire module's weights, including all sub-modules' as well.

        Args:
            immediate_only (bool, optional): If true, only immediate parameters of the module are frozen. Defaults to False.
        """
        for param in self.parameters(recurse = not immediate_only):
            param.requires_grad = False
    
    def disk_size(self) -> int:
        """Returns size of the model in bytes."""
        allparams = self.parameters(recurse=True)
        allbytes = 0
        for param in allparams:
            allbytes += (param.numel() * param.element_size())
        return allbytes
    
    def save(self, name : str = None, quiet : bool = False) -> None :
        """Saves this model's weights as a `.model` file.
        By default, files are saved @ .talos

        Args:
            name (str): filename to save it as.
        """
        if name is None: name = self.name
        final = os.path.join(modelsavesdir, name)
        dirr = os.path.dirname(final)
        if dirr: os.makedirs(os.path.dirname(final), exist_ok=True)
        save_model(self, f'{final}.model')
        if not quiet: print(f'Saved model @ `{name}.model`')
    
    
    def load(self, name : str) -> None:
        """Loads this model's weights from a `.model` file.
        By default, files are saved @ .talos

        Args:
            name (str): filename to load it from.
        """
        load_model(self, f'{os.path.join(modelsavesdir, name)}.model', strict = True)
        return self
    
    
    def forward(self, x : Tensor) -> Tensor:
        return x
        
        
        


