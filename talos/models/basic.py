
import os
from typing import Literal

import torch.nn as nn

from ..utils import talosdir, Tensor, tensor
from safetensors.torch import save_model, load_model


modelsavesdir = os.path.join(talosdir, 'models')


class TalosModule(nn.Module):
    """Better module than a simple nn.Module (which it subclasses), adding more functionality."""
    
    n = 0
    def __init__(self, name : str = None, *args, **kwargs) -> None:
        self.name = name
        self.frozen : None | Literal['all', 'immediate', 'all_but_immediate'] = None
        if name is None:
            self.name = f'module_{TalosModule.n}'
            TalosModule.n += 1
        super().__init__(*args, **kwargs)
    
    def nparams(self) -> int:
        """Returns all the params in this model."""
        n = sum(list(map(lambda x:x.numel(), self.parameters(recurse=True))))
        return n

    def freeze(self, immediate_only : bool = False):
        """Freezes the entire module's weights, including all sub-modules' as well.
        You can check frozen status using `.frozen`.

        Args:
            immediate_only (bool, optional): If true, only immediate parameters of the module are frozen. Defaults to False.
        """
        for param in self.parameters(recurse = not immediate_only):
            param.requires_grad = False
        self.frozen = 'immediate' if immediate_only else 'all'
        
        return self
    
    def unfreeze(self, immediate_only : bool = False):
        """Unfreezes the entire module's weights, including all sub-modules' as well.
        You can check frozen status using `.frozen`.

        Args:
            immediate_only (bool, optional): If true, only immediate parameters of the module are unfrozen. Defaults to False.
        """
        for param in self.parameters(recurse = not immediate_only):
            param.requires_grad = True
        
        if self.frozen is not None:
            if self.frozen == 'immediate':
                self.frozen = None
            elif self.frozen == 'all':
                if immediate_only: self.frozen = 'all_but_immediate'
                else: self.frozen = None
            else:
                if not immediate_only: self.frozen = None
        
        return self
        
    
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
        

def talosify(module : nn.Module) -> TalosModule:
    """
    > DONT USE THIS FOR NOW
    
    Converts a pytorch module to TalosModule.

    Args:
        module (nn.Module): model to convert.

    Returns:
        TalosModule: converted module.
    """
    module.__class__ = TalosModule
    return module




