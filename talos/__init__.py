"""

# Talos

A library to hold all robotics related DL implementations.

"""


from . import datapipe
from . import models
# from . import training
from . import utils
from . import layers

from .utils import *
from .datapipe import load_dataset, Dataset
from .training import train
from .models import TalosModule, talosify
from .models.basic import get_arch



