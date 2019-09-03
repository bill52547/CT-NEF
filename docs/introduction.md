# Introduction

## SRF-NEF module Architecture
We designed SEF-NEF(NEF in short) as high-level object-oriented toolbox, but it is also meet the concepts such as functional programming and JSON-API. On the other hand, we learnt some key technique from Tensorflow where we based our NEF to fasten the computing.

This document describes the system architecture that makes this combination of static typing check and other automatic tools avaaiable. It assumes that you have basic familiarity with Python programming and Function programming concepts. 

This document is for nefers who want to extend NEF in some way not supported by current APIs. To obey the principle that we made as the base of NEf would help you to use the tools we provided. We would add some low-level tools lately. This document would introduce the overall picture of NEF family and start at the APP layer. 

## `JSON-API`
The overall picture of NEF in python would be thought as a JSON-API connected network. Everything can be converted to a Json string (or a dictionary in python). For every function in Nef you can assume it accepts some dictionaries and reture another dictionary. Moreover, for any class in NEF, no matter it is functional or now, it is only a dictionary covered with some existing and useful tools. We would save the schema of all NEF generated classes as JSON template. In this view, NEF is Object-Oriented. 

## `MetaClass`
*A metaclass in Python is a class of a class that defines how a class behaves. A class is itself an instance of a metaclass. A class in Python defines how the instance of the class will behave. In order to understand metaclasses well, one needs to have prior experience working with Python classes. Before we dive deeper into metaclasses, let's get a few concepts out of the way.*

We will skip the basic `metaclass` introduce the we we used it. 

Since the instance of a `metaclass` would be a class, we would the instances of a specific `metaclass` with is `meta` name.

![Figure.1](./pic/fig1.png)

- `BaseMeta` is the most basic metaclass. It is the base `metaclass` for all objects in NEF.
- `ConfigMeta` is the metaclass for a `Config` class, who
  - do not have callable field
  - all field types are basic, which are not instances of `BaseMeta`
  - do not have `data` field. 
  - can be nested with another `Config` class instances
- `DataMeta` is the metaclass for classes who have data, literally. A `Data` class would have
  - have `data` field
  - do not have callable field
  - The `data` field is the only field that is expected to be evolved.
  - can be nested with `Config` or `Data` class instances
- `MixinMeta` is the metaclass for all classes who implement some basic methods, who
  - have only callable fields
  - each callable field is a function who accept only basic typed arguments
  - each callable field is a function who return one basic output.
- `FuncMeta` instances would transform some `Data` class instance to some others.
  - do not have `data` field
  - can be nested with `Config` or `Data` class instances
  - must have `__call__` field.
  - can have non-basic typed fields
  - each callable field accept `Data` class instances and `Config` class instances. 
  - each callable field accept `Data` class instances and `Config` class instances. 

### The `data` field
A `data` field is the only field that is expected to be evolved. If some other part of a `Data` class instance in some way, it should be redefined as a new class with `data` field. 

### Differences between a `Data` class and a `Config` class:
`Data` classes are pretty much like `Config` classes but the `data` field and field types. However they are totally different!! A `Config` class instance would not be contant and plain. It have only basic typed. A `Config` class instance would not evoled during the whole process. Actually it may work as a plain `tuple`/`dict`. That is why we implemented `__eq__` method on `Config` classes. On the other hand, a `Data` class instance would be thought as a `Variable` in tensorflow with some necessary parameters. In most of the cases, `Data` class instances are the expected arguments and return for a computing quest. And `__eq__` method is not friendly on `Data` class.

### `Mixin` Layer
A `Mixin` class would thought be the most algorithmatic heavy part. Any complicated computing part would be regarded to be implemented as a `Mixin` class. In fact, a `Mixin` class instance would be though unpack all the objects. However, unpacked objects are ugly and are not the things we want, so we start the methd names in `Mixin` class with `_`.

Another thing to be mentioned is that the `Mixin` layer is absolutely independent with `Data` layer. If you want to implement a method is a `Func` class, you can just inherit the corresponding `Mixin` class, covered with some wanted strutures. This inheriance is save since a `Mixin` class would not have regular attribute fields. It would not try to change the existing `__slots__`. In fact a `Data` class can also inherit `Mixin` class. But it is not recommended. You should always thinking if you can remove all the unproperty methods in `Data` class. 

### `Func` layer
`Func` classes are the highest-level classes. It can have anything but `data` field. A `Func` class instance would be regards are a transformer from some `Data` / `Config` class instances to other `Data` / `Config` class instances.


### Overall picture
If we would like to plot a graph with nodes and edges of all the classes we have in NEF to figure out all the dependencies:
- The `Mixin` class would not on this picture
- The `Config` classes can be hidden since they are naive. It can also perform as `Data` class
- `Data` classes are the nodes
- `Func` classes are the directed edges. 