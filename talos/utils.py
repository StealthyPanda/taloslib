

import numpy as np
import torch
import pathlib
import os



from typing import Any
from codex import log, warning, error, ok
from PIL.Image import Image


Tensor = torch.Tensor
tensor = torch.tensor

talosdir = '.talos'



def in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except Exception:
        return False



def gpu_info() -> None:
    """Prints GPU related info. Use `gpu_exists()` to get a boolean."""
    
    ndevices  = torch.cuda.device_count()
    print(f'Found {ndevices} cuda devices...')
    for each in range(ndevices):
        intname = f'cuda:{each}'
        log(
            each, f' / {intname}', '\t',
            torch.cuda.get_device_name(each), '\t',
            f'{torch.cuda.get_device_properties(each).total_memory / pow(2, 20):.2f}MB', '\t',
            f'{torch.cuda.get_device_properties(each).total_memory / pow(2, 30):.2f}GB', #'\t',
            # f'{torch.cuda.get_device_properties(each).name}',
        )


def gpu_exists(verbose : bool = True) -> bool:
    """Returns true if GPU exists on this machine. Else, false."""
    
    if torch.cuda.device_count() > 0:
        if verbose : ok('GPU(s) exist!')
        return True
    else:
        if verbose : warning('No GPUs found!')
    return False



def disk_size(module : torch.nn.Module) -> int:
    """Returns size of the model in bytes."""
    allparams = module.parameters(recurse=True)
    allbytes = 0
    for param in allparams:
        allbytes += (param.numel() * param.element_size())
    return allbytes



class ImageToTensor:
    """
    Basically like the transform `ToTensor` in `torchvision`, but much, MUCH better and way less frustrating. 
    - Supports batching, and bascially any type of image data under the sun. 
    - Drop in replacement, so you can use it literally anywhere the `torchvision`'s `ToTensor` is used. 
    - Will give you a valid `torchvision` image tensor at the end automatically. 
    - `Tensor`s are passed through *AS IS*.
    - All image data are assumed to be cv2-esque shaped \
        `(batch_dim[optional], height, width, channels[optional])` in any form (python list, \
            numpy arrays, pytorch tensors, PIL Images etc.).
    
    Related: `TensorToImage`.
    """
    def __init__(self, batched : bool = True, cv2_img : bool = False, scale : bool = True) -> None:
        self.batched : bool = batched
        self.cv2_img : bool = cv2_img
        self.scale : bool = scale
    
    def __call__(self, images : Any, batched : bool = None, cv2_img : bool = None, scale : bool = None) -> torch.Tensor:
        """Converts any image to Tensors.

        Args:
            images (Any): Literally anything representing an image.
            batched (bool, optional): Whether multiple images are being passed. Defaults to True.
            cv2_img (bool, optional): Set this to true if images are in BGR or BGRA. Defaults to False.
        """
        self.batched = batched if batched is not None else self.batched
        self.cv2_img = cv2_img if cv2_img is not None else self.cv2_img
        self.scale = scale if scale is not None else self.scale
        
        allimgs = []
        
        if type(images) == list:
            for _, each in enumerate(images):
                if isinstance(each, torch.Tensor): allimgs.append(each.numpy())
                elif isinstance(each, Image): allimgs.append(np.array(each))
                elif isinstance(each, np.ndarray): allimgs.append((each))
                else: allimgs.append(np.array(each))
            allimgs = np.array(allimgs)
            allimgs = torch.tensor(allimgs)
        
        elif isinstance(images, torch.Tensor):
            allimgs = images
        
        elif isinstance(images, np.ndarray):
            allimgs = torch.tensor(images)
        
        
        if self.batched: self.image_size = allimgs.shape[1:]
        else: self.image_size = allimgs.shape
        
        
        if len(self.image_size) == 3:
            self.channels = self.image_size[-1]
            self.image_size = self.image_size[:-1]
        else:
            self.channels = 0
        
        
        if self.channels == 0: allimgs = torch.unsqueeze(allimgs, -1)
        
        if not self.batched: allimgs = torch.unsqueeze(allimgs, 0)
        
        allimgs = torch.transpose(allimgs, 2, 3)
        allimgs = torch.transpose(allimgs, 1, 2)
        
        if self.cv2_img:
            if self.channels == 3:
                allimgs = torch.flip(allimgs, [1])
            elif self.channels == 4:
                allimgs = torch.concat((
                    torch.flip(allimgs[:, :3, :, :], [1]),
                    allimgs[:, 3:, :, :]
                ), dim = 1)
        
        if self.scale:
            allimgs /= torch.max(allimgs)
            
        return allimgs


class TensorToImage:
    """
    Inverse of `talos.ImageToTensor`. Gives you a `plt.imshow`-able image at the end. 
    Basically gives you a numpy array, with the shape `(batch_dim [optional], height, width, channels [optional])`. 
    Matches closely to the original format given, and RGB or BGR formats.
    """
    
    def __init__(self, img_to_tensor : ImageToTensor = None) -> None:
        self.img_to_tensor = img_to_tensor
    
    def __call__(self, images_tensor : torch.Tensor, img_to_tensor : ImageToTensor = None) -> torch.Tensor:
        img_to_tensor = img_to_tensor if img_to_tensor is not None else self.img_to_tensor
        
        batched = img_to_tensor.batched
        cv2_img = img_to_tensor.cv2_img
        channels = img_to_tensor.channels
        
        images = images_tensor.numpy(force=True)        
        images = np.transpose(images, (0, 2, 3, 1))
        
        
        if cv2_img:
            if channels == 3:
                images = images[:, ::-1, :, :]
            elif channels == 4:
                images = np.concat((
                    images[..., :3][..., ::-1],
                    images[..., 3:]
                ), axis = -1)
        
        return images





class FileExtractor:
    """
    Recursively extracts file paths in given directory with given extentions.
    You can use it just like a python list.
    
    Example usage:
    
    >>> txtfiles = FileExtractor()
    >>> for each in txtfiles:
    >>>     pass #do something with each 
    
    """
    def __init__(self, root_dir : str | os.PathLike = '.') -> None:
        """Recursively extracts file paths in given directory with given extentions.

        Args:
            root_dir (str | os.PathLike): directory to look in. Defaults to '.'.
        """
        self.root = pathlib.Path(root_dir)
        self.file_paths = [file for file in self.root.rglob('*') if file.is_file()]
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, *index) -> os.PathLike:
        return self.file_paths.__getitem__(*index)

class ImageExtractor(FileExtractor):
    """
    Recursively extracts image file paths in given directory.
    You can use it just like a python list.
    
    Example usage:
    
    >>> imgfiles = ImageExtractor()
    >>> for each in imgfiles:
    >>>     pass #do something with each 
    
    """
    def __init__(
            self, 
            root_dir: str | os.PathLike = '.',
            additional_image_formats : list[str] = None,
        ) -> None:
        """Recursively extracts image file paths in given directory.
            You can use it just like a python list.

        Args:
            root_dir (str | os.PathLike, optional): directory to look in. Defaults to '.'.
            additional_image_formats (list[str], optional): additional image formats to include. Default extentions are ['png', 'jpg', 'jpeg'].
        """
        super().__init__(root_dir)
        self.image_formats = [
            'png', 'jpg', 'jpeg'
        ]
        if additional_image_formats is not None: self.image_formats += additional_image_formats
        self.file_paths = list(filter(
            lambda x: os.path.basename(x).split('.')[1].lower() in self.image_formats, 
            self.file_paths
        ))