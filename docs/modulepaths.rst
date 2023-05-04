.. _module paths:

Module Paths
============

torch lets you reference individual modules in a model by a string name, given by `named_modules()`. It's often more convenient to work with these strings than object references to individual modules, as they are easier to debug and transfer between instances of a model.

nice_hooks often accepts a "module path", which is a string pattern that selects a particular set of modules by name. This allows hooks to be set of multiple modules at once.

It works as follows:

* ``"my.module.name"`` - matches a specific module with this exact name
* ``"my.module.name.*"``` - matches any module that is a child of ``my.module.name``
* ``"my.module.name.**"`` - matches any module that is a descentdent of ``my.module.name``


.. _module path indices:

Module path indices
--------------------

Module paths can also contain an index operation, e.g. ``"my.module.path[0:5]"``. This instructs the hook to report the output of the module indexed to just the range specified. Similarly, any tensor returned from the hook affects just the indexed range.

Being able to have hooks target specific subsets of a layer's output means you can work with specific neurons.

* ``"my.module.path[:5]"`` - indexes just the first 5 elements in dim 0.
* ``"my.module.path[:,-5:]"``` - indexes just the last 5 elements in dim 1. This is often necessary as the first dimension is batched.
* ``"my.module.path[*]"`` - wildcard on dim 0. Will set up a hook for each of ``"my.module.path[0]"``, ``"my.module.path[1]"``, etc
