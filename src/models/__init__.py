from .eegcnn import EEGCNN
from .eegtcn import EEGTCN
from .eegrnn import EEGRNN
from .baseline_mlp import BaselineMLP

def get_model(name, **kwargs):
    if name == "eegcnn":
        return EEGCNN(**kwargs)
    # elif name == "eegtcn":
    #     return EEGTCN(**kwargs)
    # elif name == "eegrnn":
    #     return EEGRNN(**kwargs)
    # elif name == "mlp":
    #     return BaselineMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
