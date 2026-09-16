"""LATTE: post-hoc linearization of pretrained Transformer attention."""

from .attention import RTDetrLATTEAttention
from .detr_attention import DetrLATTEAttention
from .detr_utils import replace_detr_layers
from .rtdetr_utils import replace_rtdetr_layers, select_layers_by_score

__all__ = [
    "DetrLATTEAttention",
    "RTDetrLATTEAttention",
    "replace_detr_layers",
    "replace_rtdetr_layers",
    "select_layers_by_score",
]
__version__ = "1.2.0"
