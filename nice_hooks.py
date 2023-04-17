import torch as t
from torch import nn
import functools
import re
from typing import Union
from activationcache import ActivationCache, ActivationCacheLike

def _match_path_to_re(match_path: str) -> re.Pattern:
    return re.compile(match_path.replace(".", "\\.").replace("*", "\b[^.]*\b"))

class ModuleNameMatcher:
    """Utility for describing a set of module names in a simple syntax similar to .gitignore's"""
    def __init__(self, match_paths: Union[list[str], bool]):
        if match_paths is True:
            self.positive_paths = [re.compile(".*")]
            self.negative_paths = []
        elif match_paths is False:
            self.positive_paths = []
            self.negative_paths = []
        else:
            self.positive_paths = [_match_path_to_re(p) for p in match_paths if not p.startswith("!")]
            self.negative_paths = [_match_path_to_re(p[1:]) for p in match_paths if p.startswith("!")]

    def __call__(self, str):
        return all(p.fullmatch(str) for p  in self.positive_paths) and not any(p.fullmatch(str) for p  in self.negative_paths)

def run(module: nn.Module, *args, 
        return_activations: Union[list[str], bool] = None, 
        with_activations: ActivationCacheLike = None, 
        **kwargs):
    """Runs the model, accepting some extra keyword parameters
    
    return_activations - if true, records activations as the module is run an activations cache. Returns a tuple of model output, and the activations cache.
    with_activations - if set, replaces the given activations when running the module forward. """
    cleanup: list[t.utils.hooks.RemovableHandle] = []
    if with_activations:
        # Add hook to every module
        def hook(new_value, m, i, o):
            return new_value
        for name, submodule in module.named_modules():
            if name in with_activations:
                cleanup.append(submodule.register_forward_hook(functools.partial(hook, with_activations[name])))
    if return_activations:
        # Add hook to every module
        activation_cache = {}
        matcher = ModuleNameMatcher(return_activations)
        def hook(module_name, m, i, o):
            activation_cache[module_name] = o
        for name, submodule in module.named_modules():
            if matcher(name):
                cleanup.append(submodule.register_forward_hook(functools.partial(hook, name)))
    # Actually run module
    try:
        result = module(*args, **kwargs)
    finally:
        # Cleanup
        for h in cleanup:
            h.remove()
    # Return
    if return_activations:
        return result, activation_cache
    else:
        return result
    
def patch_method_to_module(cls: type, fname: str):
    """Call on a class inheriting from nn.Module to convert a method in that module to a submodule.
    This is useful if you wish to add hooks on the function.

    This uses monkey patching, you only need call it once on a class to affect all instances.
    It must be called before creating instances of cls.
    """
    # Record the old methods we're about to patch
    old_fn = getattr(cls, fname)
    old_init = getattr(cls, "__init__")
    # Define the module itself.
    class ReplacementModule(nn.Module):
        def __init__(self, parent) -> None:
            super().__init__()
            self.parent = parent
        def forward(self, *args, **kwargs):
            return old_fn(self.parent, *args, **kwargs)
    ReplacementModule.__name__ = old_fn.__name__
    # Define replacement methods
    def repl__init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self._modules[fname] = ReplacementModule(self)
    def repl_fn(self, *args, **kwargs):
        return self._modules[fname](*args, **kwargs)
    setattr(cls, fname, repl_fn)
    setattr(cls, "__init__", repl__init__)
    pass

if __name__ == "__main__":
    class CustomModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
        def inner_calc(self, t):
            return t * 2
        def forward(self, t):
            return t + self.inner_calc(t)
    
    patch_method_to_module(CustomModule, "inner_calc")
        
    

    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        CustomModule()
    )
    result, activations = run(model, t.zeros((1,)), 
                              return_activations=True,
                              with_activations={'0': t.ones((10,))})
    print(result, activations)