import torch as t
from typing import Union, List, Tuple
import builtins
_int = builtins.int

class ActivationCache(dict[str, t.Tensor]):
    """
    A wrapper around a dictionary of cached activations from a model run.
    Supports similar operations to t.Tensor, which are usually applied elementwise.

    """

    def __add__(self, other: 'ActivationCache') -> 'ActivationCache':
        keys = frozenset(self) & frozenset(other)
        return ActivationCache({k: self[k] + other[k] for k in keys})
    
    def __sub__(self, other: 'ActivationCache') -> 'ActivationCache':
        keys = frozenset(self) & frozenset(other)
        return ActivationCache({k: self[k] - other[k] for k in keys})
    
    def to(self, device: Union[str, t.device]) -> 'ActivationCache':
        """
        Moves each tensor of the cache to a device.
        """
        return ActivationCache({k: v.to(device) for k,v in self.items()})
    
    def index(self, key:Union[None, _int, slice, t.Tensor, List, Tuple]) -> 'ActivationCache':
        """Applies t[key] for every tensor in the activation cache.
        As the tensors may have different shapes, this operation only really makes sense for
        manipulating the batch dimension."""
        return ActivationCache({k: v[key] for k, v in self.items()})

# An object castable to activation cache
ActivationCacheLike = dict[str, t.Tensor]
"""Any object that a :class:`ActivationCache` can be constructed from."""