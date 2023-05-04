.. _usage:
Usage
=====


.. _registering hooks:

Registering hooks
-----------------

nice_hooks comes with functions :func:`nice_hooks.register_forward_hook`, :func:`nice_hooks.register_forward_pre_hook`, :func:`nice_hooks.register_full_backward_hook`, :func:`nice_hooks.register_full_backward_pre_hook` which correspond to their counterparts on :class:`torch.nn.Module`.

The main differences are:
 * They are functions, not methods of Module
 * The accept an additional argument ``path``, which lets you select multiple modules of the model.
 * The hook functions are called with an additional argument path which describes the module currently

Because of the ``path`` argument, you can register hooks on your root model, and specify the particular module you are interested in as a string. 
The `path` argument is a :ref:`module path <module paths>` so can specify multiple modules with wild cards, and :ref:`index specific values <module path indices>` in the activation.

.. code-block:: python

    def my_hook(module: nn.Module, path: nice_hooks.ModulePath, args, output):
        print(output)

    nice_hooks.register_forward_hook(my_model, "path.to.module", my_hook)

Running the model
-----------------

nice_hooks comes with a method ``run`` that runs the model.

.. code-block:: python

    result = nice_hooks.run(model, *args, **kwargs)
    # equivalent to:
    # result = model(*args, **kwargs)

``run`` comes with several keyword arguments for controlling running the model:

Running the model with hooks
----------------------------

Hooks can be set on the model for the duration of a single run:

.. code-block:: python

    result = nice_hooks.run(model, *args, forward_hooks={'mod1': hook1})
    # Equivalent to
    # with nice_hooks.register_forward_hook(model, 'mod1', hook1):
    #     result = model(*args)

See :ref:`registering hooks` for details.

Recording activations
---------------------

You can get the activations associated with an evaluation of the model with:

.. code-block:: python

    result, cache = nice_hooks.run(my_model, *args, return_activations=True)

The returned :ref:`cache <activation caches>` is a dictionary with a keys for each module name, and tensor values for their output during the run.

As storing all activations occupies memory, you can also specify which modules you are interested in, using the :ref:`module path syntax <module paths>`.

.. code-block:: python

    result, cache = nice_hooks.run(my_model, *args_to_model, return_activations=["mod1.*", "mod2"])

Activation patching
-------------------

You can replace specific activations with a known value:

.. code-block:: python
    
    result = nice_hooks.run(my_model, *args, with_activations={'mod1': t.ones(5)})

This replaces the output of the module named mod1 with the given tensor. Replacing an entire layer is not often useful, so you will likely want to use a :ref:`path with an index <module path indices>`


.. code-block:: python

    result = nice_hooks.run(my_model, *args, with_activations={'mod1[:,3:5]': t.ones(2)})
