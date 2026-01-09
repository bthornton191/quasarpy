"""
quasarpy - Python wrapper for ODYSSEE CAE Quasar Engine.

Provides programmatic surrogate modeling (Kriging) for simulation data.
"""

from .quasar import Quasar, DatasetConfig, KrigingConfig
from .validation import ValidationResult

__version__ = '0.1.0'
__all__ = ['Quasar', 'DatasetConfig', 'KrigingConfig', 'ValidationResult']
