import torch as t
from torch.utils.hooks import RemovableHandle
from torch import nn
import functools
import re
from typing import Union, Callable, Tuple
from activationcache import ActivationCache, ActivationCacheLike
from dataclasses import dataclass

AnySlice = Union[None, int, slice, list, tuple, t.Tensor]
PathsLike = Union[str, list[str], bool]

def to_paths(paths: PathsLike)->list[str]:
    if paths is True:
        return ["**"]
    if paths is False:
        return []
    if isinstance(paths, str):
        return [paths]
    return paths
class RemovableHandleCollection:
    def __init__(self, handles: list[RemovableHandle]) -> None:
        self.handles = handles

    def __enter__(self):
        return self
    
    def __exit__(self):
        for h in self.handles:
            h.remove()


def _format_slice(sl):
    if isinstance(sl, slice):
            start = "" if sl.start is None else sl.start
            stop = "" if sl.stop is None else sl.stop
            if sl.step is None:
                return f"{start}:{stop}"
            else:
                return f"{start}:{stop}:{sl.step}"
    else:
        return str(sl)

@dataclass
class ModulePath:
    name: str
    slice: AnySlice

    def __iter__(self):
        return iter((self.name, self.slice))

    def __str__(self) -> str:
        if self.slice == None:
            return self.name
        elif isinstance(self.slice, tuple):
            slice_str = ",".join(_format_slice(i) for i in self.slice)
            return f"{self.name}[{slice_str}]"
        elif isinstance(self.slice, slice):
            return f"{self.name}[{_format_slice(self.slice)}]"
        else:
            return f"{self.name}[{self.slice}]"

_PATH_REGEX_BITS = {
    ".": r"\.",
    "*": r"[^\.]*",
    "**": r".*"
}

_WILDCARD = "*"

def _split_module_path(path: str) -> ModulePath:
    """Parse a string into a ModulePath. Wildcards are passed through unhanged"""
    # Parse out the slice
    m = re.search(r"\[([0-9-:\*,]+)\]$", path)
    slice_parts = None
    if m is not None:
        ii = m.group(1).split(",")
        path = path[:len(path) - len(m.group(0))]
        slice_parts = []
        for i in ii:
            m2 = re.match(r"([0-9-]*):([0-9-]*)", i)
            if m2:
                slice_parts.append(slice(
                    None if m2.group(1) == "" else int(m2.group(1)),
                    None if m2.group(2) == "" else int(m2.group(2)),
                ))
                continue
            m2 = re.match(r"[0-9-]+", i)
            if m2:
                slice_parts.append(int(i))
                continue
            if i == "*":
                slice_parts.append(_WILDCARD)
                continue
            raise Exception("Unrecognized slice part", i)
        slice_parts = tuple(slice_parts)
    return ModulePath(path, slice_parts)

def expand_module_path(model: nn.Module, path: PathsLike) -> list[(ModulePath, nn.Module)]:
    """Parses a path string to a ModulePath, then expands any wildcards in the name.
    Slice wildcards are left unchanged"""
    paths = to_paths(path)

    r = []
    for path in paths:
        unexpanded = _split_module_path(path)

        # Turn path into a regex:
        # .  -> \.
        # *  -> [^\*]*
        # ** -> .*
        # everything else to literals
        repath = ""
        for item in re.split("(\.|\*\*|\*)", unexpanded.name):
            repath += _PATH_REGEX_BITS.get(item, re.escape(item))
        repath = re.compile(repath)


        # Find modules that match path
        for name, mod in model.named_modules():
            if not repath.fullmatch(name):
                continue
            r.append((ModulePath(name, unexpanded.slice), mod))
    return r

def _is_wild_slice(sl: AnySlice) -> bool:
    """Wild slices are those that may return multiple values in expand_slice"""
    return isinstance(sl, tuple) and any(i == _WILDCARD for i in sl)

def _expand_slice(tt: t.tensor, sl: AnySlice) -> list[Tuple[t.tensor, AnySlice]]:
    """Computes tt[sl]. If sl has wildcard references, expand them
    according to the shape of tt"""
    if sl is None:
        return [(tt, None)]
    elif isinstance(sl, tuple):
        combinations = []
        for i, item in enumerate(sl):
            if item == _WILDCARD:
                combinations.append(range(tt.shape[i]))
            else:
                combinations.append([item])
        from itertools import product
        return [(tt[expanded_sl], expanded_sl) for expanded_sl in product(*combinations)]
    else:
        return [(tt, sl)]

def _regroup(iter, key_fn, value_fn=None):
    """Unordred group by."""
    # Am i stupid, why doesn't python have this function?
    d = {}
    for i in iter:
        k = key_fn(i)
        v = value_fn(i) if value_fn is not None else i
        if k not in d:
            d[k] = []
        d[k].append(v)
    return d

def _do_fwd_hook(expanded_paths: list[Tuple[ModulePath, nn.Module]], hook: Callable) -> RemovableHandleCollection:
    """Like register_forward_hook, but for the results of expand_module_path
    Handles wildcard slices."""
    paths_by: dict[nn.Module, list[ModulePath]] = _regroup(expanded_paths, lambda p: p[1], lambda p: p[0])
    handles = []
    for module, path_tuples in paths_by.items():
        def inner_hook(pt, mod, i, o):
            orig_o = o
            for name, sl in pt:
                new_o = o
                for tt, sl2 in _expand_slice(o, sl):
                    update = hook(mod, ModulePath(name, sl2) , i, tt)
                    if update is not None:
                        if sl2 is None:
                            new_o = update
                        else:
                            new_o = new_o.index_put(sl2, update)
                o = new_o
            return None if o is orig_o else o

        handle = module.register_forward_hook(functools.partial(inner_hook, path_tuples))
        handles.append(handle)
    return RemovableHandleCollection(handles)

def register_forward_hook(module: nn.Module, path: PathsLike, hook: Callable) -> RemovableHandleCollection:
    expanded = expand_module_path(module, path)
    return _do_fwd_hook(expanded, hook)

def run(module: nn.Module, *args, 
        return_activations: PathsLike = None, 
        with_activations: ActivationCacheLike = None,
        **kwargs):
    """Runs the model, accepting some extra keyword parameters for various behaviours
    
    return_activations - if true, records activations as the module is run an activations cache. Returns a tuple of model output, and the activations cache.
    with_activations - if set, replaces the given activations when running the module forward.
    with_forwared_hooks - if sets, temporarily registers the forward hooks for just this run """
    cleanup: list[RemovableHandle] = []

    if with_activations:
        def with_activation_hook(new_value, m, p, i, o):
            return new_value
        for k, v in with_activations.items():
            # TODO: pre-hook?
            cleanup.extend(register_forward_hook(module, k, functools.partial(with_activation_hook, v)).handles)

    if return_activations:
        return_activations = to_paths(return_activations)
        activation_cache = ActivationCache()
        def with_return_hook(m, p, i, o):
            activation_cache[str(p)] = o
        for k in return_activations:
            cleanup.extend(register_forward_hook(module, k, with_return_hook).handles)
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
                              return_activations=["0[0:3]", "1[*]", "3.*"],
                              with_activations={'0': t.ones((10,))})
    print(result, activations)