"""NTN channel generation; the public entry point is channel_generator.HybridChannel."""

from .channel_generator import HybridChannel, generate_channel

__all__ = ["HybridChannel", "generate_channel"]
