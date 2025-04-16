
import torch
import torch.nn as nn

from .models import TalosModule



class LoRALinear(TalosModule):
    """Implements Linear layer with a Low RAnk weight matrix.
    This will decrease no. of weights, but increase training time for convergence."""
    
    def __init__(
            self, 
            in_size : int, out_size : int, lrank : int = None,
            bias : bool = True,
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
        
        self.bias = bias
        if bias:
            self.c = nn.Parameter(torch.zeros( (out_size, ) )) 
        
    
    def forward(self, x):
        W = self.a @ self.b
        x = (x @ W)
        if self.bias: x = x + self.c
        return x


