
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import TalosModule
from ..utils import Tensor

from .archs import AttentionFFN, ParallelAttention



class AttentionMultiDimBlock(TalosModule):
    """Multi-headed attention on Multi-Dimensional features (2D, 3D...), followed by MLP."""
    
    def __init__(
            self, 
            context_size : int,
            embed_shape : tuple[int, ...],
            nheads : int,
            efactor : int = 4,
            dropout : float = 0.2,
            name: str = None,
            *args, **kwargs
        ) -> None:
        """Attention -> MLP

        Args:
            context_size (int): size of context window
            embed_shape (tuple[int, ...]): shape of input and output features.
            nheads (int): no. of parallel heads
            efactor (int, optional): size of hidden state in MLP is `efactor * embed_total_size`. Defaults to 4.
            dropout (float, optional): dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.embed_shape = embed_shape
        self.nheads = nheads
        self.total_size = reduce(lambda x, y: x*y, embed_shape, 1)
        self.dropout = dropout
        self.context_size = context_size
        self.efactor = efactor
        
        self.attention = ParallelAttention(
            in_channels=self.total_size,
            context_size=context_size,
            head_size=self.total_size,
            nheads=nheads,
            droprate=self.dropout,
        )
        self.mlp = AttentionFFN(self.total_size, efactor=efactor, droprate=self.dropout)
        
    
    def forward(self, x: Tensor) -> Tensor:
        x # b, t, *embed_shape
        
        x = torch.flatten(x, 2) # b, t, total_size
        
        x = self.attention(x) # b, t, total_size
        x = self.mlp(x) # b, t, total_size
        
        x = torch.reshape(x, (x.shape[0], x.shape[1], *self.embed_shape)) # b, t, *embed_shape
        
        return x
        
    


