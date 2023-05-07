.. nice_hooks documentation master file, created by
   sphinx-quickstart on Thu May  4 14:15:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nice_hooks's documentation!
======================================



A python package for making working with torch hooks easier. 
You can bind hooks to a modules using a :ref:`flexible pattern string <module paths>` that can target multiple modules/:ref:`neurons <module path indices>` and it contains utilities for working with :ref:`activation caches <activation caches>`.

Many of the ideas are based on the ``TransformerLens`` library, but written so it can work with *any* pytorch model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage.rst
   modulepaths.rst
   activationcache.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
