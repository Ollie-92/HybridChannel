"""Compatibility imports; use channel_generator.py for the implementation."""
from .channel_generator import *
from . import channel_generator as _implementation

def __getattr__(name):
    return getattr(_implementation, name)
