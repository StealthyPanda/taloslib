
import os
import warnings
from typing import Any, Callable
from codex import ok, warning, error

import numpy as np

import torch

from ..utils import Tensor
from .basic import TalosModule, talosdir



class Modification:
    """
    Similar to `transforms` for tensors, this is for pytorch modules. 
    Use this to compose repeatable modifications to model architectures.
    
    Example:
    >>> import talos
    >>> from talos.models.modify import Compose, ReplaceWithIdentity
    
    >>> model = talos.models.hub.IntelDepthModel().fetch_model(empty=True)
    >>> mods = Compose(ReplaceWithIdentity(-1), ReplaceWithIdentity(-2))
    >>> model = mods(model) # modified model
    >>> model = model.load('saved_model') # If the model was saved with modifications, it must be loaded after doing those same modifications.
    
    """
    def forward(self,  module : TalosModule) -> TalosModule:
        """Define your module modification logic here.

        Args:
            module (TalosModule): input module.

        Returns:
            TalosModule: modified module.
        """
        return module
    def __call__(self, module : TalosModule) -> TalosModule:
        return self.forward(module)

class ReplaceWith(Modification):
    def __init__(self, new_module : TalosModule, index : int = None, layer_name : str = None) -> None:
        """Replaces layer at given location by the `new_module`. Location can be given 
        either through `index` or `layer_name`, but not both.

        Args:
            new_module (TalosModule): module to replace with.
            index (int, optional): index location of the layer. Defaults to None.
            layer_name (str, optional): fully qualified named location of the module. Defaults to None.
        """
        super().__init__()
        self.new_module = new_module
        self.index = index
        self.layer_name = layer_name
        assert not ((index is None) and (layer_name is None)), (
            f'Need one of `index` or `layer_name`!'
        )
        assert not (( not (index is None) ) and ( not (layer_name is None) )), (
            f'Cannot use both `index` and `layer_name`!'
        )
    
    def forward(self, module: TalosModule) -> TalosModule:
        module
        
        if self.index is not None:
            module.replace_layer(self.index, self.new_module)
        if self.layer_name is not None:
            module.replace_layer_by_name(self.layer_name, self.new_module)
        
        return module


class ReplaceWithIdentity(ReplaceWith):
    def __init__(self, index: int = None, layer_name: str = None) -> None:
        super().__init__(torch.nn.Identity(), index, layer_name)

class Compose(Modification):
    def __init__(self, *mods : list[Modification]) -> None:
        """Composes modifications one after the other, similar to `nn.Sequential`."""
        super().__init__()
        self.mods = mods
    
    def forward(self, module: TalosModule) -> TalosModule:
        module
        
        for each in self.mods:
            module = each(module)
        
        return module

