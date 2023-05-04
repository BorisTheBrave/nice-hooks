# nice_hooks

A python package for making working with torch hooks easier. You can bind hooks to a modules using a [flexible pattern string](#module-paths) and it contains utilities for working with [activation caches](#activationcache).

Many of the ideas are based on the `TransformerLens` library, but written so it can work with *any* pytorch model.
