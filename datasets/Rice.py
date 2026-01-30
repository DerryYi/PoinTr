from .PCNDataset import PCN
from .build import DATASETS

@DATASETS.register_module()
class Rice(PCN):
    pass
