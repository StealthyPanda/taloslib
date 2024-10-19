
import os
import warnings
from typing import Any, Callable

import torch

from ..utils import Tensor
from .basic import TalosModule, talosdir



pretrained_dir = os.path.join(talosdir, 'pretrained', 'torchhub')

torch.hub.set_dir(pretrained_dir)
warnings.filterwarnings('ignore')

print(
    'Torch Hub models can be accessed @ https://pytorch.org/hub/'
)




class TorchHubModel(TalosModule):
    
    def __init__(
            self,
            url : str,
            model_name : str,
            utils_name : str | None = None, 
            name: str = None, 
            *args, **kwargs
        ) -> None:
        """Loads a model from `torch.hub`. Model is loaded lazily; call `.fetch_model()` to actually 
        load the model.

        Args:
            url (str): url to load from
            model_name (str): name of the model
            utils_name (str | None, optional): Optional utils to load from. Defaults to None.
        """
        
        super().__init__(name, *args, **kwargs)
        
        self.url = url
        self.model_name = model_name
        self.utils_name = utils_name
        
        self.model : TalosModule = None
        self.utils = None
        
        self.fetched = False
    
    def fetch_model(self):
        print(f'Fetching model... `{self.model_name}`')
        self.model = torch.hub.load(
            self.url, self.model_name, pretrained=True, trust_repo=True
        )
        if self.utils_name is not None:
            self.utils = torch.hub.load(self.url, self.utils_name, trust_repo=True)
        self.fetched = True
        
        return self
    
    def forward(self, *inputs, **kinputs) -> Any:
        if not self.fetched: self.fetch_model()
        return self.model(*inputs, **kinputs)





