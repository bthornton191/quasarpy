"""
quasarpy - Python wrapper for ODYSSEE CAE Quasar Engine.

Provides programmatic surrogate modeling (Kriging) for simulation data.
"""

from .quasar import Quasar, DatasetConfig, KrigingConfig
from .validation import ValidationResult, LearningCurveResult
from .optimization import (
    ObjectiveConfig,
    AggregatedObjectiveConfig,
    ConstraintConfig,
    OptimizationResult,
)

__version__ = '0.6.0'
__all__ = [
    'Quasar',
    'DatasetConfig',
    'KrigingConfig',
    'ValidationResult',
    'LearningCurveResult',
    'ObjectiveConfig',
    'AggregatedObjectiveConfig',
    'ConstraintConfig',
    'OptimizationResult',
]
