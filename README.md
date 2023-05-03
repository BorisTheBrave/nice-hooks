# nice_hooks

A python package for making working with torch hooks easier.

Many of the ideas are based on the `TransformerLens` library, but written so it can work with *any* pytorch model.

## Usage

### Registering hooks

### Running the model

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

TODO: The current behaviour only patches entire layers, which is hardly helpful.
Maybe we should support some sort of slicing? "my_module[:, 0]"


## Concepts

### Module paths

torch lets you reference individual modules in a model by a string name, given by `named_modules()`. It's often more convenient to work with these strings than object references to individual modules, as they are easier to debug and transfer between instances of a model.

nice_hooks often accepts a "module path", which is a string pattern that selects a particular set of modules by name. This allows hooks to be set of multiple modules at once.

 It works as follows:

* `"my.module.name"` - matches a specific module with this exact name
* `"my.module.name.*"` - matches any module that is a child of `my.module.name`
* `"my.module.name.**`" - matches any module that is a descentdent of `my.module.name`

#### Module path indices

Module paths can also contain an index operation, e.g. `"my.module.path[0:5]"`. This instructs the hook to report the output of the module indexed to just the range specified. Similarly, any tensor returned from the hook affects just the indexed range.

Being able to have hooks target specific subsets of a layer's output means you can

* `"my.module.path[:5]"` - indexes just the first 5 elements in dim 0.
* `"my.module.path[:,-5:]"` - indexes just the last 5 elements in dim 1. This is often necessary as the first dimension is batched.
* `"my.module.path[*]"` - wildcard on dim 0. Will set up a hook for each of `"my.module.path[0]"`, `"my.module.path[1]"`, etc

### ActivationCache

Another key concept is `ActivationCache`, which is a dictionary of tensors, with keys corresponding to names of modules.

Activation caches are used to represent a collection of intermediate output values from a given run of a model. They are very useful for interpretability - the activations can be inspected for particular behaviour, or copied between inference runs.

nice_hooks generally returns a class `ActivationCache` that has some useful methds on it, but nice_hooks function arguments will accept any dictionary of strings to tensors (called a `ActivationCacheLike`).
