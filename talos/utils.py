import torch

from codex import log, warning, error, ok



Tensor = torch.Tensor
tensor = torch.tensor



def gpu_info() -> None:
    """Prints GPU related info. Use `gpu_exists()` to get a boolean."""
    
    ndevices  = torch.cuda.device_count()
    print(f'Found {ndevices} cuda devices...')
    for each in range(ndevices):
        log(
            each, '\t',
            torch.cuda.get_device_name(each), '\t',
            f'{torch.cuda.get_device_properties(0).total_memory / pow(2, 20):.2f}MB', '\t',
            f'{torch.cuda.get_device_properties(0).total_memory / pow(2, 30):.2f}GB',
        )


def gpu_exists() -> bool:
    """Returns true if GPU exists on this machine. Else, false."""
    
    if torch.cuda.device_count() > 0:
        ok('GPU(s) exist!')
        return True
    else:
        warning('No GPUs found!')
    return False


