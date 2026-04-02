"""
Model components for Biomedical Vision-Language Model

Components:
- vision_encoder: BiomedCLIP for medical image feature extraction
- language_model: BioMistral-7B with LoRA for efficient fine-tuning
- projector: Multi-layer network for feature alignment
- vlm: Complete Vision-Language Model
"""

from .vision_encoder import BiomedCLIPEncoder
from .language_model import BioMistralLM
from .projector import MultiLayerProjector, LinearProjector, IdentityProjector
from .vlm import BiomedVLM

__all__ = [
    'BiomedCLIPEncoder',
    'BioMistralLM',
    'MultiLayerProjector',
    'LinearProjector',
    'IdentityProjector',
    'BiomedVLM'
]
