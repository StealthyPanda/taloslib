
import os

import torch

from ..utils import Tensor
from .basic import TalosModule, talosdir

from transformers import AutoModel, AutoImageProcessor, pipeline



pretrained_dir = os.path.join(talosdir, 'pretrained', 'huggingface')



class PreTrainedBaseImageModel(TalosModule):
    """Wrapper for using Hugging Face pretrained base image models.
    
    > ** Note: the model is loaded lazily; won't be loaded/downloaded until the
    first time it is actually used. **
    
    Use `last_hidden_state_only = True` when passing to the call of this model, to get only the last layer's output.
    """
    
    def __init__(self, hf_model : str, name: str = None, *args, **kwargs) -> None:
        """
        Args:
            hf_model (str): model name on huggingface (eg. `facebook/detr-resnet-50`)
        """
        super().__init__(name, *args, **kwargs)
        
        self.model_name = hf_model
        self.model = None
        self.input_processor = None
    
    def load_model(self):
        if self.model is not None: return
        
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir = pretrained_dir, output_hidden_states=True
        )
        self.input_processor = AutoImageProcessor.from_pretrained(
            self.model_name, cache_dir = pretrained_dir
        )
        
    
    def forward(self, x : Tensor, last_hidden_state_only : bool = True, **additional_args) -> Tensor:
        self.load_model()
        y = self.model(**self.input_processor(
            images = x, return_tensors='pt', **additional_args
        ))
        if last_hidden_state_only: return y.last_hidden_state
        else: return y


class PreTrainedImageModel(TalosModule):
    """Wrapper for using Hugging Face pretrained end-to-end image models.
    
    > **Note: the model is loaded lazily; won't be loaded/downloaded until the
    first time it is actually used.**
    """
    
    def __init__(
            self, 
            hf_model : str, task : str = 'object-detection', device : str = 'cuda',
            name: str = None, *args, **kwargs
        ) -> None:
        """
        Args:
            hf_model (str): model name on huggingface (eg. `facebook/detr-resnet-50`)
        """
        super().__init__(name, *args, **kwargs)
        
        self.model_name = hf_model
        self.task = task
        self.device = device
        
        self.pipe = None
        self.model = None
    
    def load_model(self):
        if self.model is not None: return
        
        self.pipe = pipeline(
            task=self.task, model=self.model_name, device=self.device,
            cache_dir=pretrained_dir
        )
        self.model = self.pipe.model
        self.input_processor = self.pipe.image_processor
        self.post_processor = self.pipe.postprocess
        
    
    def forward(self, x : Tensor, **additional_args) -> Tensor:
        self.load_model()
        y = self.input_processor(images=x, return_tensors='pt', **additional_args)
        y = y.to(self.device)
        y = self.model(**y)
        return y


        
        

