"""SAM 3.1 uses its isolated PyTorch environment and the common mask protocol."""
from efficient_sam_client import EfficientSAMClient


class SAM31Client(EfficientSAMClient):
    experiment = 'sam31'
    label = 'SAM 3.1'
