"""
AI Bias Analysis Framework

A comprehensive toolkit for detecting, measuring, and mitigating bias in AI models.
"""

__version__ = "1.0.0"
__author__ = "Mustafiz Ahmed"
__email__ = "aimjetkhalifa10@gmail.com"

from src.data.data_generator import BiasedDataGenerator
from src.metrics.fairness_metrics import FairnessMetrics
from src.mitigation.bias_mitigation import (
    BiasPreprocessor,
    FairnessConstrainedModel,
    BiasPostprocessor,
    BiasMitigationPipeline
)
from src.visualization.visualizer import BiasVisualizer

__all__ = [
    'BiasedDataGenerator',
    'FairnessMetrics',
    'BiasPreprocessor',
    'FairnessConstrainedModel',
    'BiasPostprocessor',
    'BiasMitigationPipeline',
    'BiasVisualizer',
]