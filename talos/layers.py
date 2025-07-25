
from typing import Callable

import torch
import torch.nn as nn

from .models import TalosModule



class LoRALinear(TalosModule):
    """Implements Linear layer with a Low RAnk weight matrix.
    This will decrease no. of weights, but increase training time for convergence."""
    
    def __init__(
            self, 
            in_size : int, out_size : int, lrank : int = None,
            use_bias : bool = True,
            name = None, *args, **kwargs
        ):
        """

        Args:
            in_size (int): size of input features
            out_size (int): size of output features
            lrank (int, optional): inner low rank dimension. Defaults to `min(in_size, out_size) / 4`.
            bias (bool, optional): whether to add a bias. Defaults to True.
        """
        super().__init__(name, *args, **kwargs)
        
        if lrank is None:
            lrank = max(min(in_size, out_size) // 4, 1)
        
        self.a = nn.Parameter(torch.randn( (in_size, lrank) ))
        self.b = nn.Parameter(torch.randn( (lrank, out_size) ))
        
        self.bias = use_bias
        if self.bias:
            self.c = nn.Parameter(torch.zeros( (out_size, ) )) 
        
    
    def forward(self, x):
        W = self.a @ self.b
        x = (x @ W)
        if self.bias: x = x + self.c
        return x


class Lambda(TalosModule):
    """Turns a lambda function into a module. Useful for dropping into `nn.Sequential`."""
    def __init__(self, fn : Callable, name = None, *args, **kwargs):
        """Turns a lambda function into a module. 
        Simply passes inputs to given function, and returns output.
        Useful for dropping into `nn.Sequential`. 
        Can be used for many things like hooks too.

        Args:
            fn (Callable): lambda expression, or any function.
        """
        super().__init__(name, *args, **kwargs)
        
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)