.. _Activation Caches:

ActivationCache
===============


Another key concept is :class:`nice_hooks.ActivationCache`, which is a dictionary of tensors, with keys corresponding to names of modules.

Activation caches are used to represent a collection of intermediate output values from a given run of a model. They are very useful for interpretability - the activations can be inspected for particular behaviour, or copied between inference runs.

nice_hooks generally returns a class :class:`nice_hooks.ActivationCache` that has some useful methds on it, but nice_hooks function arguments will accept any dictionary of strings to tensors (called a ``ActivationCacheLike``).