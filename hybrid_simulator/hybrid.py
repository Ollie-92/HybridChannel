"""Compatibility imports; use hybrid_model.py for the implementation."""
from .hybrid_model import *
from . import hybrid_model as _implementation

def __getattr__(name):
    return getattr(_implementation, name)
