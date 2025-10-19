
import os
from typing import Literal
from deprecated import deprecated

import torch.nn as nn

from ..utils import talosdir, Tensor, tensor
from safetensors.torch import save_model, load_model


modelsavesdir = os.path.join(talosdir, 'models')


def _get_attr_name_for_submod(model : nn.Module, submod : nn.Module) -> str | None:
    for name, layer in model.named_children():
        if layer is submod: return name


def _get_hierarchy(module : nn.Module, recurse : bool = False) -> list[str]:
    """Gets string representation of the hierarchical representation of this module. 
    Best used for finding what is in the damn module.

    Args:
        module (nn.Module): module to traverse.

    Returns:
        list[str]: info about the module.
    """
    name = module._get_name()
    device = None
    if hasattr(module, 'device'):
        device = module.device
    info = f'{name}({type(module).__name__}{", " + str(device) if device is not None else ""})'
    children = list(module.named_children())
    if len(children) == 0: return [info]
    
    info = [info + ':']
    
    for name, child in children:
        if not recurse: 
            sub = [
                f'    {name}({type(module).__name__}{", " + str(device) if device is not None else ""})'
            ]
        else:
            sub = _get_hierarchy(child, recurse=True)
            sub = list(map(lambda x: f'    |{x[1]}', enumerate(sub)))
            sub[0] = f'    {name} -> {sub[0][5:]}'
        info += sub

    return info

def get_arch(module : nn.Module, recurse : bool = False) -> list[str]:
    """Gets string representation of the hierarchical representation of this module. 
    Best used for finding what is in the damn module.

    Args:
        module (nn.Module): module to traverse.

    Returns:
        list[str]: info about the module.
    """
    return '\n'.join(_get_hierarchy(module, recurse))


def move_to(module : nn.Module, device : str):
    module.to(device)
    if type(module) == nn.ModuleList:
        for submod in module : move_to(submod, device)
    for child in module.children():
        move_to(child, device)
    


class TalosModule(nn.Module):
    """Better module than a simple nn.Module (which it subclasses), adding more functionality."""
    
    n = 0
    def __init__(self, name : str = None, *args, **kwargs) -> None:
        self.name = name
        self.device = 'cpu'
        self.frozen : None | Literal['all', 'immediate', 'all_but_immediate'] = None
        if name is None:
            self.name = f'module_{TalosModule.n}'
            TalosModule.n += 1
        super().__init__(*args, **kwargs)
    
    def to(self, device, *args, **kargs):
        self.device = device
        for child in self.children():
            # child.to(device)
            # if type(child) == nn.ModuleList:
            #     for each in 
            move_to(child, device)
        return super().to(device, *args, **kargs)
    
    def arch(self, recurse : bool = False) -> str:
        return get_arch(self, recurse=recurse)
    
    def __str__(self) -> str:
        return self.arch()
    
    def nparams(self) -> int:
        """Returns all the params in this model."""
        n = sum(list(map(lambda x:x.numel(), self.parameters(recurse=True))))
        return n
    
    
    def ntrainable_params(self) -> int:
        n = sum(list(map(lambda x:x.numel(), filter(lambda x:x.requires_grad, self.parameters(recurse=True)) )))
        return n
    
    @property
    def size_info(self) -> str:
        """Get size info about this model."""
        nparams = self.nparams() / 1e6
        ntrainable = self.ntrainable_params() / 1e6
        disksize = self.disk_size()/1e6
        return (
            f'{nparams:.3f}M params, {ntrainable:.3f}M trainable params ({ntrainable * 100 / nparams:.1f}%), {disksize:.3f}MB total in memory'
        )

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

        
    
    def replace_layer_by_name(self, layer : str, new_module):
        """Replaces a layer with another, given its fully qualified name.

        Args:
            layer (str): fully qualified name of the layer
            new_module (TalosModule): module to replace with.
        """
        parent_module = self
        parent_name = layer.split('.')
        layer_name = parent_name[-1]
        parent_name = '.'.join(parent_name[:-1])
        
        if parent_name: parent_module = self.get_submodule(parent_name)
        
        setattr(parent_module, layer_name, new_module)
        
        
        return self
    
    def replace_layer(self, layer : int, new_module):
        """Replaces a layer with another by index.

        Args:
            layer (int): index of the layer to replace.
            new_module (TalosModule): module to replace with.
        """
        return self.replace_layer_by_name(self.layer_names[layer], new_module)
    
    @property
    def layer_names(self) -> list[str]:
        """Gives you all the fully qualified layer names of each of the layers in this module."""
        return list(map(lambda x:x[0], self.named_modules()))
    
    
    def forward(self, x : Tensor) -> Tensor:
        return x
    
    def tune(self, *args, **kwargs):
        """Sets the model in fine-tune mode.
        Optionally takes arguments. Override this in subclasses to do custom finetuning logic. 
        By default does nothing.
        """
        return self


# def get_children_names(module : nn.Module | TalosModule) -> list[str]:
#     return list(map(lambda x:x[0], module.named_children()))
    



@deprecated(reason='Not working yet')
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




