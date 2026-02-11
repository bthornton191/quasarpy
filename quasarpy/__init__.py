"""
quasarpy - Python wrapper for ODYSSEE CAE Quasar Engine.

Provides programmatic surrogate modeling (Kriging) for simulation data.
"""

from .quasar import Quasar, DatasetConfig, KrigingConfig
from .validation import ValidationResult, LearningCurveResult
from .optimization import ObjectiveConfig, ConstraintConfig, OptimizationResult

__version__ = '0.5.0'
__all__ = [
    'Quasar',
    'DatasetConfig',
    'KrigingConfig',
    'ValidationResult',
    'LearningCurveResult',
    'ObjectiveConfig',
    'ConstraintConfig',
    'OptimizationResult'
]
