
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import TalosModule
from ..utils import Tensor


class FFN(TalosModule):
    """Good Old flat, Feed Forward Networks."""
    
    def __init__(self, layers : list[int], activation = F.relu, in_channels : int = None) -> None:
        """
        Args:
            layers (list[int]): list of ints, no. of cells in each layer. 
            activation (function(tensor) -> tensor, optional): can be a single function or a list of functions, for each layer. Defaults to F.relu.
        """
        super().__init__()
        
        self.layers = nn.ModuleList(
            [nn.LazyLinear(layers[0]) if in_channels is None else nn.Linear(in_channels, layers[0])] + 
            [nn.Linear(layers[i - 1], each) for i, each in enumerate(layers) if i >= 1]
        )
        self.outsize = layers[-1]
        
        self.activation = activation
        if type(activation) != list:
            self.activation = [activation for _ in self.layers]
        
    def forward(self, x : Tensor) -> Tensor:
        x # (b, incells)
        
        for each, act in zip(self.layers, self.activation):
            x = each(x)
            x = act(x)
        
        x # (b, outsize)
        
        return x



class UNetBlock3D(TalosModule):
    """The UNetBlock, basically a Convolution, Activation and then Pooling."""
    def __init__(
            self,
            c_in : int, c_out : int,
            kernel_size : tuple[int, int, int] = (3, 3, 3),
            pool_size : tuple[int, int, int] = (2, 2, 2),
            res_connection : bool = False
        ):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out if c_out is not None else 2 * c_in
        self.kernel_size = kernel_size
        self.rescon = res_connection
        
        self.conv = nn.Conv3d(
            in_channels=c_in,
            out_channels= c_out,
            kernel_size=kernel_size,
            padding='same'
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(pool_size)
        
    
    def forward(self, x : Tensor) -> Tensor:
        x # (b, c_in, t, x, y)
        
        original = None
        if self.rescon:
            original = x
        
        x = self.conv(x) # (b, c_out, t, x, y)
        x = self.relu(x) # (b, c_out, t, x, y)
        
        if self.rescon:
            x = x + original
        
        x = self.pool(x) # (b, c_out, t // 2, x // 2, y // 2)
        
        return x





class AttentionHead(TalosModule):
    """Single Attention head."""
    
    def __init__(
            self,
            in_channels : int,
            context_size : int,
            head_size : int,
            droprate : float = 0.2,
        ) -> None:

        super().__init__()
        
        self.key = nn.Linear(in_channels, head_size, bias = False)
        self.query = nn.Linear(in_channels, head_size, bias = False)
        self.value = nn.Linear(in_channels, head_size, bias = False)
        
        self.t = context_size
        self.hs = head_size
        
        self.register_buffer('mask', torch.tril(torch.ones(self.t, self.t)))
        self.dropout = nn.Dropout(droprate)
    
    
    def forward(self, x):
        x # b t c
        
        b, t, c = x.shape
        
        k = self.key(x) # b t hs
        q = self.query(x) # b t hs
        
        weights = q @ k.transpose(-2, -1) * (c ** -0.5) # b t hs @ b hs t = b t t
        weights = weights.masked_fill(self.mask[:t, :t] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        
        v = self.value(x) # b t hs
        out = weights @ v # b t t @ b t hs = b t hs
        
        return out
        


class ParallelAttentionHead(TalosModule):
    """Parallel Attention head."""
    
    def __init__(
            self,
            in_channels : int,
            context_size : int,
            head_size : int,
            nheads : int,
            droprate : float = 0.2
        ) -> None:
        
        super().__init__()
        
        self.heads = []
        for _ in range(nheads):
            self.heads.append(
                AttentionHead(
                    in_channels=in_channels,
                    context_size=context_size,
                    head_size = int(head_size / nheads),
                    droprate = droprate,
                )
            )
        self.heads = nn.ModuleList(self.heads)
        self.dropout = nn.Dropout(droprate)
        self.projector = nn.Linear(head_size, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        
    
    def forward(self, x):
        x # b t c
        out = torch.concat(list(map( lambda head : head(x), self.heads )), dim = -1)
        out = self.dropout(self.projector(out)) # b t c
        return out




class ParallelAttention(TalosModule):
    def __init__(
            self, 
            in_channels : int,
            context_size : int,
            head_size : int,
            nheads : int,
            droprate : float = 0.2,
            name: str = None, 
            *args, **kwargs
        ) -> None:
        super().__init__(name, *args, **kwargs)
        
        self.aheads = nn.ModuleList([
            AttentionHead(
                in_channels=in_channels,
                context_size=context_size,
                head_size= head_size // nheads,
                droprate = droprate,
            ) for _ in range(nheads)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x
        subheads = list(map(lambda xhead:xhead(x), self.aheads))
        head = torch.concat(subheads, -1)
        return head





class AttentionFFN(FFN):
    """Attention FFN."""
    def __init__(self, in_channels : int, efactor : int = 4, droprate : float = 0.2):
        super().__init__([in_channels * efactor, in_channels], [F.relu, lambda x:x], in_channels)
        self.dropper = nn.Dropout(droprate)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.dropper(x)
        
        return x


class Block(TalosModule):
    """A self attention block."""
    
    def __init__(
            self,
            in_channels : int,
            context_size : int,
            head_size : int,
            nheads : int,
            droprate : float = 0.2,
        ) -> None:
        super().__init__()
        
        self.att = ParallelAttention(
            in_channels = in_channels,
            context_size= context_size,
            head_size = head_size,
            nheads = nheads,
            droprate = droprate,
        )
        
        self.attln = nn.LayerNorm(in_channels)
        
        self.ffn = AttentionFFN(
            in_channels = in_channels,
            efactor = 4,
            droprate = droprate,
        )
        
        self.ffnln = nn.LayerNorm(in_channels)
        
    
    def forward(self, x):
        x #b t c
        x = x + self.att(self.attln(x))
        x = x + self.ffn(self.ffnln(x))
        return x




class Transformer(nn.Module):
    """Complete Self attention transformer"""
    def __init__(
            self, 
            vsize : int,
            d_embed : int,
            context_size : int,
            nblocks : int,
            head_size : int,
            nheads : int,
            droprate : float = 0.2,
        ) -> None:
        super().__init__()
        
        self.token_embedder = nn.Embedding(vsize, d_embed)
        self.position_embedder = nn.Embedding(context_size, d_embed)
        
        self.blocks = [
            Block(
                in_channels = d_embed,
                context_size = context_size,
                head_size = head_size,
                nheads = nheads,
                droprate = droprate,
            ) for _ in range(nblocks)
        ]
        
        self.blocks = nn.Sequential(*self.blocks)
        
        self.norm = nn.LayerNorm(d_embed)
        self.unembedder= nn.Linear(d_embed, vsize)
        
        self.context_size = context_size
        
    

    
    
    def forward(self, inputs : Tensor, debug : bool = False) -> Tensor:
        b, t = inputs.shape
        
        tokens = self.token_embedder(inputs) + self.position_embedder(torch.arange(t, device=self.device))
        if debug: print(tokens.shape)
        tokens = self.blocks(tokens)
        if debug: print(tokens.shape)
        tokens = self.norm(tokens)
        if debug: print(tokens.shape)
        tokens = self.unembedder(tokens)
        if debug: print(tokens.shape)
        return tokens
    
    
    def generate_tokens(self, seed : Tensor, max_tokens : int = 2000) -> Tensor:
        
        tokens = seed # b n
        
        for each in range(max_tokens):
            print(f'Generating token {each+1}/{max_tokens}...', end = '\r')
            
            tokensub = tokens[:, -self.context_size:] # b t (last t)
            
            logits = self(tokensub) # b t vsize
            
            logits = logits[:, -1, :] # b -1 vsize = b vsize
            
            probs = F.softmax(logits, dim = -1) # b vsize
            
            pred = torch.multinomial(probs, num_samples = 1) # b 1
            
            tokens = torch.concat((tokens, pred), dim = 1)
            
        
        print()
        return tokens
    



