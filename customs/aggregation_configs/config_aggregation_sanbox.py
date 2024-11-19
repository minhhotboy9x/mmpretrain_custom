# aggregate all configurations for the project
from mmengine.config import read_base

# Use read_base to include base configurations
with read_base():
    # Import base configurations
    from mmpretrain.configs._base_.default_runtime import *
    from mmpretrain.configs._base_.schedules.imagenet_bs256 import *
    
    # Import your custom model and dataset configurations inside read_base
    from ..configs.model import *  # Custom model configuration
    from ..configs.dataset import *  # Custom dataset configuration
