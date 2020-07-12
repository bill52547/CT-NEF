.. _Quickstart:

QuickStart
==========
It assumes you already have `nefct` installed, if you do not, head over to the
:ref:`Installation` section.

Assuming you have access to `/mnt/gluster/Techpi/` which is the official gluster directory of
*Tech-Pi* company.

A minimal application
---------------------

A minimal SRF-NEF application looks something like this:

.. code-block:: python

    %matplotlib inline
    import nefct as nef
    img = nef.load('/mnt/gluster/Techpi/SRF-NEF/resource/demo/helloworld/image.hdf5')
    if nef.utils.is_notebook():
        img.imshow()
    print('helloworld!')

if you see *helloworld!* printed, good job! You are now a **Nefer**!


A simple reconstruction
-----------------------

The application above only imports package `nefct` and illustrates a slice of existing image (if
you are in `jupyter notebook` environment). Let's move on to reconstructing projection. We will
use **brain16** phantom to do this reconstruction. Before this, please check if you have at least
one GPU mounted and CUDA toolkit installed.

.. code-block:: python

    %matplotlib inline
    import nefct as nef
    mlem = nef.load('/mnt/gluster/Techpi/SRF-NEF/resource/demo/simple_recon/mlem.hdf5')
    listmode = nef.load('/mnt/gluster/Techpi/SRF-NEF/resource/demo/simple_recon/listmode.hdf5')
    img = mlem(listmode)
    if nef.utils.is_notebook():
        img.imshow()
    print('helloworld, again!')


If you see **helloworld, again!** printed, well done! This is a whole reconstruction of
**brain16**.

So what dis that code do?

1. First we imported the package `nefct` as `nef`.
2. Next we loaded a `mlem` and `listmode` which is a function and a data,     separately. Yes. We saved a function in memory. In this way, a existing    `mlem` reconstruction program (function) can be repeatedly used on         different listmode. The `listmode` is the data to be reconstructed.
3. We then apply the `mlem` function on data `listmode` to do the              reconstruction. This reconstruction will last 10 iterations.
4. an `img` will come out being displayed. It should be exactly same with that in `helloworld`.

During we importing `nefct`, we

1. checking the package is well installed,
2. write `TYPE_BIND` with necessary components,
3. import `nef.load` we would use later.

The `TYPE_BIND` was initialized as a empty dictionary and been updated later. It records the bind
bewteen a `type` and a `string` to parse from string to corresponding type. Common types in
`nefct` package have been fulfilled in `TYPE_BIND`. Once new types are defined as a `DataClass`,
the `TYPE_BIND` was updated automatically. You can also added it into `TYPE_BIND` as below, to
allow automatically parsing.

.. code-block:: python

    from nefct.typing import TYPE_BIND
    TYPE_BIND.update({'your-type': your-type})

A comprehensive Reconstruction
------------------------------

Somtimes, a full reconstruction need to be built personally from system descriptions. We take a **PET** system as example.

.. code-block:: python

    import numpy as np
    import h5py
    %matplotlib inline

    import nefct as nef

    block = nef.Block(np.array([20, 33.4, 33.4]), np.array([1, 10, 10]))
    scanner = nef.PetCylindricalScanner(99.0, 119.0, 1, 16, 0, block)
    shape = np.array([90, 90, 10])
    center = np.array([0.,0.,0.])
    size = np.array([180., 180., 33.4])

    with h5py.File('/mnt/gluster/Techpi/brain16/recon/data/cylinder/small_cylinder_air_trans.h5', 'r') as fin:
        fst = np.array(fin['listmode_data']['fst'])
        snd = np.array(fin['listmode_data']['snd'])

    listmode = nef.Listmode.from_lors(nef.Lors.from_fst_snd(fst, snd)).compress(scanner)

    projector = nef.Projector('siddon','gpu')
    bprojector = nef.BackProjector(shape, center, size, 'siddon', 'gpu')

    print('emaping...')
    emap = nef.EmapMlem.from_scanner(scanner, bprojector, 'full')

    print('reconstructing with MLEM...')
    mlem = nef.Mlem(10, projector, bprojector, emap)
    img = mlem(listmode)
    if nef.utils.is_notebook():
        img.imshow()
    print('helloworld, again again!')
A sample to build of DataClass and FuncClass
--------------------------------------------

With `nefct` package, new functions or data can be added as esay as breath. We provide some examples here about how to build a new `DataClass` or `FuncClass` with `nefct` and their features.

Example 1. Build a new DataClass
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Before build a new `DataClass`, we need to clarify what is a `DataClass`. Similar with it been implemented in `Python 3.7`_, The `DataClass` is a way of automating the generation of boiler-plate code for classes which store multiple properties. A DataClass is built with a `dataclass` decorator over class definition.

.. _Python 3.7: https://hackernoon.com/a-brief-tour-of-python-3-7-data-classes-22ee5e046517

.. code-block:: python

    from nefct.typing import dataclass

    @dataclass
    class SimpleClass(object):
        field_0: str

    from nefct.typing import TYPE_BIND
    TYPE_BIND.update({'SimpleClass': SimpleClass})

    simple_obj = SimpleClass('hello')

1. `dataclass` decorator, for decorating a data class
2. The `field` method for configuration fields.
3. Update the `TYPE_BIND` in `nefct.typing` to bind the string (class name) with this class. This step help the `io` to parse the classname to corresponding class.

(TODO: refering Python 3.7, implement the dataclass from official site.)

Note, all the fields of a `DataClass` are frozen. We prefer the users to build a new instance but changing one. To build a new instance from the current, we implemented a `replace` method in `DataClass`, replacing fields with values from changes. Some more features can be found in the following examples.

.. code-block:: python

    # simple_obj.field_0 = 'world' -> FrozenInstanceError()
    simple_obj2 = simple_obj.replace(field_0 = 'world')

    _dict = simple_obj.as_dict()
    # _dict = {'field_0': 'hello'}



Example 2. Build a new `FuncClass`
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

As the name `FuncClass` says, a FuncClass is a 'Function Class'. It is a special `DataClass` which has `__call__` method. In this way, it can behave as a function, to be called with. It works similar with `DataClass`,being built with a decorator `funcclass`.

.. code-block:: python

    from nefct.typing import funcclass

    @funcclass
    class PrintClass(object):
        prefix: str

        def __call__(self, string1, string2):
            return self.prefix + ' ' + string1 + ' ' + string2

    from nefct.typing import TYPE_BIND
    TYPE_BIND.update({'PrintClass': PrintClass})

    hello_sth = PrintClass('hello')
    print(hello_sth('world,', 'my friend'))
    # hello world, my frien

Some more features have been implemented on `FuncClass`, with examples shown below.

.. code-block:: python

    hello_sth_my_friend = hello_sth.currying('my friend')
    print(hello_sth_my_friend('moto,'))
    # hello moto, my friend

    the = PrintClass('the')
    what_world = the.currying('world')

    print((hello_sth_my_friend @ what_world)('cruel'))
    # hello the cruel world my friend
    # _sth = hello_sth_my_friend @ what_world('cruel')
    # _sth() -> AttributeError

We firstly implemented `FuncClass.currying` to provide function currying_. Currying provides a way for working with functions that take multiple arguments, and using them in frameworks where functions might take only one argument. Our function currying is not exactly same with its strict definition. We binded the rest arguments as fields in a new `FuncClass` instance for regarding the first argument as the only one. So we call the first argument of a `FuncClass` instance as **key argument**. This is a nautral thought! A function should accept one argument and return one output. For sure, this preference is not strict in `nefct` package.

Function currying make it more clear to do function compositions. An regular function composition is implemented by nesting, even multiple nesting. For example `f(g(a, b), c)`.  What if composition with more functions is needed. It would be a disaster After currying them, we can simplicit it to `f1(g1(a)) = f1 @ g1(a)`. We used @ in `nefct` package to present function composition.

As we mentioned above, we can save a function with package `nefct`. This is implemented with `FuncClass`. In this way, some useful function can be saved and loaded when needed.

.. code-block:: python

    from nefct import save, load
    save('./what_world.hdf5', what_world)
    what_world2 = load('./what_world.hdf5')
    print(what_world2('better'))
    # the better world

.. _currying: https://en.wikipedia.org/wiki/Currying

Example 3. A special DataClass
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

There is a special DataClass in `nefct`, DataClass with `data`, literally. it is a DataClass with field `data`. We defined this kind of class because most of the fields in a DataClass are mostly unwanted changable. For example, if you defined an BankAccount class, the only field that changes usually is your balance. The other fields, for example, `name` and `address` would rarelly change. We say the balance is the data in your `BankAccount`.

.. code-block:: python

    from nefct.typing import dataclass
    @dataclass
    class BankAccount:
        data: int
        name: str
        address: str

        @property
        def balance(self):
            return self.data

We used data to represent your balance in your BankAccount for some reason. We can define a `@property` function `balance` to return the balance(data). We used a special field name `data` because we have defined some math operators on DataClass, which works on field data by default. For example, you have two account in this stupid bank (because it doesn't recognize bank account with account numbers) and you want to combine them to a new account.

.. code-block:: python

    account1 = BankAccount(100, 'Minghao', 'Tech-Pi')
    account2 = BankAccount(200, 'Minghao', 'Tech-Pi')
    new_account = account1 + account2
    # new_account.balance = 300

And you can do some business and double your BankAccount balance.

.. code-block:: python

    im_rich = new_account * 2
    # im_rich.balance = 600

All these ufunc operators have been implemented in `numpy.ndarray` and `numba.cuda`. With the second one, it will work on GPUs.




































123
