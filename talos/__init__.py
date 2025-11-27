"""

# Talos

A library to hold all robotics related DL implementations.

"""


from . import datapipe
from . import models
from . import utils
from . import layers
from . import training


from .utils import *
from .datapipe import load_dataset, Dataset
from .models import TalosModule, talosify
from .models.basic import get_arch



