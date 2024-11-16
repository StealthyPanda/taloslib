
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
                this attention block is in decoder part of an encoder \
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
                this attention block is in decoder part of an encoder \
                decoder architecure (cross attention part). Defaults to True.
            droprate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.size_per_head = head_size // nheads
        
        self.heads = nn.ModuleList([
            Attention(
                in_size=in_size,
                head_size= self.size_per_head,
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
        
        x = list(map(
            lambda head: head[1](
                    x, 
                    keys[:, :, head[0]  * self.size_per_head : (head[0] + 1)  * self.size_per_head] if keys is not None else None,
                    queries[:, :, head[0]  * self.size_per_head : (head[0] + 1)  * self.size_per_head] if queries is not None else None, 
                    values[:, :, head[0]  * self.size_per_head : (head[0] + 1)  * self.size_per_head] if values is not None else None
                ), 
            enumerate(self.heads)
        ))
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
                this attention block is in decoder part of an encoder \
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



class AttentionMultiDimFFN(TalosModule):
    
    def __init__(
            self, 
            in_shape : tuple[int, ...],
            efactor : int = 4,
            name: str = None, *args, **kwargs
        ) -> None:
        """FFN with a single hidden layer, with a size of `efactor * input size`. \
            Hidden state also goes through ReLU activation. Output shape is same as input shape.

        Args:
            in_shape (tuple[int, ...]): shape of inputs (not including batch and context dims)
            efactor (int, optional): factor to multiply by, to get hidden state size. Defaults to 4.
        """
        super().__init__(name, *args, **kwargs)
        
        self.insize = reduce(lambda x,y : x*y, in_shape, 1)
        self.inshape = in_shape
        
        self.ffn = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(self.insize, int(self.insize * efactor)),
            nn.ReLU(),
            nn.Linear(int(self.insize * efactor), self.insize),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x
        
        inshape = x.shape
        
        x = self.ffn(x)
        x = torch.reshape(x, inshape)
        
        return x




class EncoderDecoderTransformerBlock(TalosModule):
    """Implements an encoder-decoder block as shown in:
    - https://arxiv.org/pdf/1706.03762
    - https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    """
    def __init__(
            self,
            embed_shape : tuple[int, ...],
            context_size_encoder : int,  context_size_decoder : int, 
            efactor : int = 4,
            nheads : int = 4, droprate : float = 0.2,
            all_unmasked : bool = False,
            name: str = None, *args, **kwargs
        ) -> None:
        """Creates a single encoder-decoder stackable block of a transformer. \
            Contains LayerNorms and Residual connections. Implemented closely from \
            https://arxiv.org/pdf/1706.03762 & https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
        
        
        Args:
            embed_shape (tuple[int, ...]): shape of the input and output embeddings
            context_size_encoder (int): size of encoder context window
            context_size_decoder (int): size of decoder context window
            efactor (int, optional): `efactor` for the `AttentionMultiDimFFN`. Defaults to 4.
            nheads (int, optional): no. of parallel heads. Defaults to 4.
            droprate (float, optional): dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.encoder_self_attention = AttentionMultiDim(
            embed_shape=embed_shape,
            head_shape=embed_shape,
            context_size=context_size_encoder,
            nheads=nheads, masked=False, droprate=droprate,
            name = f'{self.name}_encoder_self_attention'
        )
        self.encoder_cross_attention = AttentionMultiDim(
            embed_shape=embed_shape,
            head_shape=embed_shape,
            context_size=context_size_encoder,
            nheads=nheads, masked=False, droprate=droprate,
            name = f'{self.name}_encoder_cross_attention'
        )
        self.decoder_self_attention = AttentionMultiDim(
            embed_shape=embed_shape,
            head_shape=embed_shape,
            context_size=context_size_decoder,
            nheads=nheads, masked=not all_unmasked, droprate=droprate,
            name = f'{self.name}_decoder_self_attention'
        )
        self.decoder_cross_attention = AttentionMultiDim(
            embed_shape=embed_shape,
            head_shape=embed_shape,
            context_size=context_size_decoder,
            nheads=nheads, masked=False, droprate=droprate,
            name = f'{self.name}_decoder_cross_attention'
        )
        self.encoder_mlp = AttentionMultiDimFFN(
            in_shape=embed_shape, efactor=efactor,
            name=f'{self.name}_encoder_ffn'
        )
        self.decoder_mlp = AttentionMultiDimFFN(
            in_shape=embed_shape, efactor=efactor,
            name=f'{self.name}_decoder_ffn'
        )
        
        
        self.norm1 = nn.LayerNorm(embed_shape)
        self.norm2 = nn.LayerNorm(embed_shape)
        self.norm3 = nn.LayerNorm(embed_shape)
        self.norm4 = nn.LayerNorm(embed_shape)
        self.norm5 = nn.LayerNorm(embed_shape)
        
        
    
    def forward(self, x_enc: Tensor, x_dec : Tensor) -> Tensor:
        x_enc 
        x_dec
        
        enc_self_att, _, _, _ = self.encoder_self_attention(x_enc)
        enc_self_att = self.norm1(enc_self_att + x_enc)
        
        enc_out = self.encoder_mlp(enc_self_att)
        enc_out = self.norm2(enc_out + enc_self_att)
        
        _, k, _, v = self.encoder_cross_attention(enc_out)
        
        dec_self_att, _, _, _ = self.decoder_self_attention(x_dec)
        dec_self_att = self.norm3(dec_self_att + x_dec)
        
        dec_cross_att, _, _, _ = self.decoder_cross_attention(dec_self_att, keys=k, values=v)
        dec_cross_att = self.norm4(dec_cross_att + dec_self_att)
        
        dec_out = self.decoder_mlp(dec_cross_att)
        dec_out = self.norm5(dec_out + dec_cross_att)
        
        return dec_out
        


class SelfAttentionTransformerBlock(TalosModule):
    
    def __init__(
            self, 
            embed_shape : tuple[int, ...],
            context_size : int,
            efactor : int = 4,
            nheads : int = 4, droprate : float = 0.2,
            masked : bool = True,
            name: str = None, *args, **kwargs
        ) -> None:
        """Self attention transformer block. Basically what you would use in LLMs etc. \
            Contains LayerNorms and Residual connections. Stackable.

        Args:
            embed_shape (tuple[int, ...]): shape of embeddings
            context_size (int): size of context window
            efactor (int, optional): factor to multiply embed size with to get hidden state size. Defaults to 4.
            nheads (int, optional): no. of parallel heads. Defaults to 4.
            droprate (float, optional): dropout rate. Defaults to 0.2.
        """
        super().__init__(name, *args, **kwargs)
        
        self.attention = AttentionMultiDim(
            embed_shape=embed_shape,
            head_shape=embed_shape,
            context_size=context_size,
            nheads=nheads, masked=masked, droprate=droprate,
            name=f'{self.name}_attention_block'
        )
        
        self.ffn = AttentionMultiDimFFN(
            in_shape=embed_shape, efactor=efactor,
            name=f'{self.name}_ffn'
        )
        
        self.norm1 = nn.LayerNorm(embed_shape)
        self.norm2 = nn.LayerNorm(embed_shape)
        self.dropper = nn.Dropout(droprate)
    
    
    def forward(self, x: Tensor) -> Tensor:
        x
        
        att, _, _, _ = self.attention(x)
        x = self.norm1(att + x)
        
        out = self.ffn(x)
        x = self.norm2(out + x)
        
        x = self.dropper(x)
        
        return x
        
        
        


