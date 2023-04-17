# nice_hooks

This module it easy to work with intermediate activations of a pytorch model. It's based on `TransformerLens`, but written so it can work with *any* pytorch model. Any functionality specific to transformers is omitted.

The key concept is an `ActivationCache`, which is a dictionary of tensors, with keys corresponding to submodules of the model, as they appear in `named_modules`.

## Usage

### Recording activations

You can get the activations associated with an evaluation of the model with:

```python
import nice_hooks
result, cache = nice_hooks.run(my_model, *args_to_model, return_activations=True)
```

As activations are large, you can also specify which submodules you are interested in.

```python
result, cache = nice_hooks.run(my_model, *args_to_model, return_activations=["mod1.*", "mod2"])
```

### Working with activations

`ActivationCache` is just a dictionary of strings to torch tensors, but it does come with a few extra utility methods such as adding/subtracting caches.

### Activation patching

TODO