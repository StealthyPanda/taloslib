
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import TalosModule
from ..utils import Tensor



def position_embedding(context_size : int, d_embed : int, N : int = 1e+3) -> Tensor:
    """Gives you positional embeddings for 1-D embeddings.

    Args:
        context_size (int): Size of context window.
        d_embed (int): Size of embeddings.
        N (int, optional): The N parameter as described in the original paper. Defaults to 1e+3.
    """
    embs = torch.arange(0, d_embed // 2, 1)
    embs = torch.unsqueeze(embs, -1)
    embs = torch.tile(embs, (1,2))
    embs = torch.flatten(embs)
    embs = torch.unsqueeze(embs, -1)
    embs = torch.tile(embs, (1, context_size))
    
    embs = embs * (2 / d_embed)
    embs = torch.pow(N, -embs)
    
    for each in range(context_size):
        embs[:, each] *= each
    
    for each in range(d_embed):
        if each % 2: embs[each] = torch.cos(embs[each])
        else: embs[each] = torch.sin(embs[each])
    
    return embs


class Attention(TalosModule):
    """Simple attention mechanism. Implements both self and cross attention. Works on 1D embeddings only, \
        assuming incoming tensors of shape `(batch size, context size, embedding size)`."""
    def __init__(
            self, 
            in_size : int,
            head_size : int,
            context_size : int,
            masked : bool = True,
            droprate : float = 0.2,
            name: str = None, 
            *args, **kwargs
        ) -> None:
        """Creates a single attention block. Only attention, no MLP or anything else. Simply applies attention,
        and a dropout. That's it.

        Args:
            in_size (int): Size of embedding features (must be 1D, so size is an integer value)
            head_size (int): Size of head
            context_size (int): Size of context window.
            masked (bool, optional): Whether this attention is masked or not. \
                You want this to be True if using self-attention (encoder) only architecture, or if \
                this attention block is in decoder part of and encoder \
                decoder architecure (cross attention part). Defaults to True.
            droprate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.key = nn.Linear(in_size, head_size, bias = False)
        self.query = nn.Linear(in_size, head_size, bias = False)
        self.value = nn.Linear(in_size, head_size, bias = False)

        self.dropper = nn.Dropout(droprate)
        
        self.t = context_size
        self.insize = in_size
        
        self.register_buffer('mask', torch.tril(torch.ones(self.t, self.t)))
        self.masked = masked
    
    
    def forward(
            self, 
            x: Tensor, 
            keys : Tensor = None, 
            queries : Tensor = None,
            values : Tensor = None,
        ) -> Tensor:
        x # b, t, in_size
        
        t = x.shape[1]
        
        if keys is None: keys = self.key(x) # b, t, hs
        if queries is None: queries = self.query(x) # b, t, hs
        if values is None: values = self.value(x) # b, t, hs
        
        weights : Tensor = queries @ keys.transpose(-2, -1) * (self.insize ** -0.5) # b, t, hs @ b, hs, t = b, t, t
        if self.masked:
            weights = weights.masked_fill(self.mask[:t, :t] == 0, float('-inf')) # b, t, t
        
        weights = F.softmax(weights, dim = -1) # b, t, t
        weights = self.dropper(weights)
        
        output = weights @ values # b, t, t @ b, t, hs = b, t, hs
        
        return output, keys, queries, values




class ParallelAttention(TalosModule):
    """Parallel Attention, simply builds on `Attention` by concatenating in the end."""
    def __init__(
            self, 
            in_size : int,
            head_size : int,
            context_size : int,
            nheads : int = 4,
            masked : bool = True,
            droprate : float = 0.2,
            name: str = None, 
            *args, **kwargs
        ) -> None:
        """Creates a single multi-headed parallel attention block. \
            Only attention, no MLP or anything else. Simply applies attention, and a dropout. That's it.

        Args:
            in_size (int): Size of embedding features (must be 1D, so size is an integer value)
            head_size (int): Size of head
            context_size (int): Size of context window.
            nheads (int) : No. of heads to use. Defaults to 4.
            masked (bool, optional): Whether this attention is masked or not. \
                You want this to be True if using self-attention (encoder) only architecture, or if \
                this attention block is in decoder part of and encoder \
                decoder architecure (cross attention part). Defaults to True.
            droprate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.heads = nn.ModuleList([
            Attention(
                in_size=in_size,
                head_size= head_size // nheads,
                context_size=context_size,
                masked=masked,
                droprate=droprate,
                name=f'{self.name}_sub_head_{each}'
            ) for each in range(nheads)
        ])
        
        
    
    def forward(
            self, 
            x: Tensor, 
            keys : Tensor = None, queries : Tensor = None, values : Tensor = None
        ) -> Tensor:
        x #b, t, insize
        
        x = list(map(lambda head: head(x, keys, queries, values), self.heads))
        if keys is None:
            keys = list(map(lambda each : each[1], x))
            keys = torch.concat(keys, dim = -1)
        if queries is None:
            queries = list(map(lambda each : each[2], x))
            queries = torch.concat(queries, dim = -1)
        if values is None:
            values = list(map(lambda each : each[3], x))
            values = torch.concat(values, dim = -1)
        
        x = list(map(lambda each : each[0], x))
        x = torch.concat(x, dim = -1) # b, t, hs
        
        return x, keys, queries, values
        





class AttentionMultiDim(TalosModule):
    """Multi-headed parallel attention on Multi-Dimensional features (2D, 3D...)."""
    
    def __init__(
            self, 
            embed_shape : tuple[int, ...],
            head_shape : tuple[int, ...],
            context_size : int,
            nheads : int = 4,
            masked : bool = True,
            droprate : float = 0.2,
            name: str = None,
            *args, **kwargs
        ) -> None:
        """Multi-headed parallel attention on Multi-Dimensional features (2D, 3D...). \
            Assuming incoming tensor is of shape `(batch size, context size, emb_dim 1, emb_dim 2, ...)`

        Args:
            embed_shape (tuple[int, ...]): Shape of embedding features.
            head_shape (tuple[int, ...]): Shape of head.
            context_size (int): Size of context window.
            nheads (int) : No. of heads to use (make sure this is a divisor of total head size). Defaults to 4.
            masked (bool, optional): Whether this attention is masked or not. \
                You want this to be True if using self-attention (encoder) only architecture, or if \
                this attention block is in decoder part of and encoder \
                decoder architecure (cross attention part). Defaults to True.
            droprate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.embed_shape = embed_shape
        self.head_shape = head_shape
        self.total_size = reduce(lambda x, y: x*y, embed_shape, 1)
        self.head_size = reduce(lambda x, y: x*y, head_shape, 1)
        
        self.attention = ParallelAttention(
            in_size=self.total_size,
            head_size=self.head_size,
            context_size=context_size,
            masked=masked,
            nheads=nheads,
            droprate=droprate,
            name=f'{self.name}_parallel_attention_block'
        )
        
    
    def forward(
            self, 
            x: Tensor, 
            keys : Tensor = None, queries : Tensor = None, values : Tensor = None
        ) -> Tensor:
        x # b, t, *embed_shape
        
        x = torch.flatten(x, 2) # b, t, total_size
        if keys is not None: keys = torch.flatten(keys, 2) # b, t, head_size
        if queries is not None: queries = torch.flatten(queries, 2) # b, t, head_size
        if values is not None: values = torch.flatten(values, 2) # b, t, head_size
        
        x, k, q, v = self.attention(x, keys, queries, values) # b, t, head_size
        
        x = torch.reshape(x, (x.shape[0], x.shape[1], *self.head_shape)) # b, t, *head_shape
        k = torch.reshape(k, (k.shape[0], k.shape[1], *self.head_shape)) # b, t, *head_shape
        q = torch.reshape(q, (q.shape[0], q.shape[1], *self.head_shape)) # b, t, *head_shape
        v = torch.reshape(v, (v.shape[0], v.shape[1], *self.head_shape)) # b, t, *head_shape
        
        return x, k, q, v
