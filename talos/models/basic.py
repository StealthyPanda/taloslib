
import torch.nn as nn


class TalosModule(nn.Module):
    """Better module than a simple nn.Module (which it subclasses), adding more functionality."""
    
    def nparams(self) -> int:
        """Returns all the params in this model."""
        n = sum(list(map(lambda x:x.numel(), self.parameters(recurse=True))))
        return n



