# nice_hooks

A python package for making working with torch hooks easier. You can bind hooks to a modules using a [flexible pattern string](#module-paths) and it contains utilities for working with [activation caches](#activationcache).

Many of the ideas are based on the `TransformerLens` library, but written so it can work with *any* pytorch model.

## Usage

### Registering hooks

nice_hooks comes with functions `register_forward_hook`, `register_forward_pre_hook`, `register_full_backward_hook`, `register_full_backward_pre_hook` which correspond to their counterparts on `torch.nn.Module`.

The main differences are:
 * They are functions, not methods of Module
 * The accept an additional argument `path`, which lets you select multiple modules of the model.
 * The hook functions are called with an additional argument path which describes the module currently

Because of the `path` argument, you can register hooks on your root model, and specify the particular module you are interested in as a string. The `path` argument is a [module path](#module-paths) so can specify multiple modules with wild cards, and [index specific values](#module-path-indices) in the activation.

```python
def my_hook(module: nn.Module, path: nice_hooks.ModulePath, args, output):
    print(output)

nice_hooks.register_forward_hook(my_model, "path.to.module", my_hook)
```

### Running the model

nice_hooks comes with a method `run` that runs the model.

```python
result = nice_hooks.run(model, *args, **kwargs)
# equivalent to:
# result = model(*args, **kwargs)
```

`run` comes with several keyword arguments for controlling running the model:

### Running the model with hooks

Hooks can be set on the model for the duration of a single run:

```python
result = nice_hooks.run(model, *args, forward_hooks={'mod1': hook1})
# Equivalent to
# with nice_hooks.register_forward_hook(model, 'mod1', hook1):
#     result = model(*args)
```

See [Registering Hooks](#registering-hooks) for details.

### Recording activations

You can get the activations associated with an evaluation of the model with:

```python
import nice_hooks
result, cache = nice_hooks.run(my_model, *args, return_activations=True)
```

The returned [cache](#activationcache) is a dictionary with a keys for each module name, and tensor values for their output during the run.

As storing all activations occupies memory, you can also specify which modules you are interested in, using the [module path syntax](#module-paths)

```python
result, cache = nice_hooks.run(my_model, *args_to_model, return_activations=["mod1.*", "mod2"])
```

### Activation patching

You can replace specific activations with a known value:

```python
result = nice_hooks.run(my_model, *args, with_activations={'mod1': t.ones(5)})
```

This replaces the output of the module named mod1 with the given tensor. Replacing an entire layer is not often useful, so you will likely want to use a [path with an index](#module-path-indices)


```python
result = nice_hooks.run(my_model, *args, with_activations={'mod1[:,3:5]': t.ones(2)})
```

### `patch_method_to_module`

TODO


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
