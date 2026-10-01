"""Paged inference primitives; CUDA compilation is lazy and opt-in."""
from ._allocator import BlockPool
__all__ = ["BlockPool"]
