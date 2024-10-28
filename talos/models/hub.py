
import os
import warnings
from typing import Any, Callable
from codex import ok, warning, error

import numpy as np

import torch
from torchvision import transforms

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
    
    def fetch_model(self, empty : bool = False):
        """Fetches the model, either with or without weights.

        Args:
            empty (bool, optional): If true, only model architecture is loaded and not weights. Defaults to False.
        """
        warning(f'Fetching model... `{self.model_name}`')
        self.model = torch.hub.load(
            self.url, self.model_name, pretrained = not empty, trust_repo=True
        )
        if self.utils_name is not None:
            self.utils = torch.hub.load(self.url, self.utils_name, trust_repo=True)
        self.fetched = True
        ok(f'Model `{self.model_name}` fetched!')
        return self
    
    def forward(self, *inputs, **kinputs) -> Any:
        if not self.fetched: self.fetch_model()
        return self.model(*inputs, **kinputs)



class IntelDepthModel(TorchHubModel):
    """Intel Depth Model MiDas, which can get depth info from a single monocular RGB image."""
    def __init__(self, name: str = None, *args, **kwargs) -> None:
        super().__init__(
            "intel-isl/MiDaS", 'MiDaS_small', 'transforms',
            name, *args, **kwargs
        )
        self.custom_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # MiDaS normalization values
                std=[0.229, 0.224, 0.225]
            ),
        ])

    
    def fetch_model(self, empty: bool = False):
        return super().fetch_model(empty).eval().freeze()
    
    def forward(self, image : np.ndarray) -> Any:
        x = image
        
        # x = self.utils.small_transform(x).to(self.device)
        x = torch.stack(list(map(lambda img:self.custom_transform(img), x)))
        
        x = x.to(self.device)
        x = self.model(x)
        
        return x


class IntelDepthFeatureExtractor(TalosModule):
    """Uses the Intel Depth model MiDas for feature extraction. 
    Features are of shape `(batch size, 128, 128)`."""
    def __init__(self, name: str = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        
        self.extractor = IntelDepthModel()

    def load(self, name: str = None, empty : bool = False) -> None:
        """Loads the feature extractor from either saved model or fetching from original source.

        Args:
            name (str, optional): model name to load from. Defaults to fetching from hub.
            empty (bool, optional): if true simply loads arch, no weights. Defaults to False.
        """
        if empty:
            self.extractor.fetch_model(empty=True)
            self.extractor.model.fc = torch.nn.Identity()
            return
        if name is None:
            self.extractor.fetch_model()
            self.extractor.model.scratch.output_conv = torch.nn.Identity()
            return
        self.extractor.fetch_model(empty=True)
        self.extractor.model.scratch.output_conv = torch.nn.Identity()
        return super().load(name)
    
    def forward(self, images: Tensor) -> Tensor:
        """Returns features.

        Args:
            images (Tensor): must be of shape (batch size, height, width, channels)

        Returns:
            Tensor: features of shape `(batch size, 64, 128, 128)`
        """
        x = self.extractor(images) # b, n, h, w
        
        # x = torch.transpose(x, 1, 2) # b, h, n, w
        # x = torch.transpose(x, 2, 3) # b, h, w, n
        
        return x



class ResNetNVIDIA(TorchHubModel):
    """Loads the base ResNet NVIDIA model, with its necssary transforms and stuff.
    Input is cv2 images, not PIL.
    """
    def __init__(self, name: str = None, *args, **kwargs) -> None:
        super().__init__(
            "NVIDIA/DeepLearningExamples:torchhub", 'nvidia_resnet50',
            'nvidia_convnets_processing_utils',
            name, *args, **kwargs
        )
        self.trans = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])
    
    def fetch_model(self, empty: bool = False):
        return super().fetch_model(empty).eval().freeze()
    
    def forward(self, image) -> Any:
        x = image
        
        x = torch.stack(list(map(lambda img:self.trans(img), x)))
        
        x = self.model(x)
        
        return x


class NVIDIAResNetFeatureExtractor(TalosModule):
    """Uses the NVIDIA's resnet50 model for feature extraction. 
    Features are of shape `(batch size, 2048)`."""
    def __init__(self, name: str = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        
        self.extractor = ResNetNVIDIA()
        
    
    def load(self, name: str = None, empty : bool = False) -> None:
        """Loads the feature extractor from either saved model or fetching from original source.

        Args:
            name (str, optional): model name to load from. Defaults to fetching from hub.
            empty (bool, optional): if true simply loads arch, no weights. Defaults to False.
        """
        if empty:
            self.extractor.fetch_model(empty=True)
            self.extractor.model.fc = torch.nn.Identity()
            return
        if name is None:
            self.extractor.fetch_model()
            self.extractor.model.fc = torch.nn.Identity()
            return
        self.extractor.fetch_model(empty=True)
        self.extractor.model.fc = torch.nn.Identity()
        return super().load(name)
        
    
    def forward(self, images: Tensor) -> Tensor:
        """Returns features.

        Args:
            images (Tensor): must be of shape (batch size, height, width, channels)

        Returns:
            Tensor: features of shape `(batch size, 2048)`
        """
        return self.extractor(images)