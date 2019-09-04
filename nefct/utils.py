# encoding: utf-8
'''
nefct.utils
~~~~~~~~~~~~

This module provides utility functions that are used within SRF-NEF
that are alose useful for extenel comsumptions.
'''

import hashlib
import os
import platform
import re
import sys

import tqdm as tqdm_

__all__ = (
    'is_notebook', 'tqdm', '_eps', '_small', '_tiny', '_huge', '_pi', 'main_path', 'separator',
    'declare_eager_execution', 'clear_gpu')


def is_notebook():
    '''check if the current environment is `ipython`/ `notebook`
    '''
    return 'ipykernel' in sys.modules


is_ipython = is_notebook


def tqdm(*args, **kwargs):
    '''same as tqdm.tqdm
    Automatically switch between `tqdm.tqdm` and `tqdm.tqdm_notebook` accoding to the runtime
    environment.
    '''
    if is_notebook():
        return tqdm_.tqdm_notebook(*args, **kwargs)
    else:
        return tqdm_.tqdm(*args, **kwargs)


_eps = 1e-8

_small = 1e-4

_tiny = 1e-8

_huge = 1e8

_pi = 3.14159265358979323846264338

if 'Windows' in platform.system():
    separator = '\\'
else:
    separator = '/'

main_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + separator


def convert_Camal_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def convert_snake_to_Camel(name: str) -> str:
    out = ''
    for ele in name.split('_'):
        out += ele.capitalize()
    return out


def get_hash_of_timestamp():
    import time
    m = hashlib.sha256()
    timestamps = time.time()
    m.update(str(timestamps).encode('utf-8'))
    return m.hexdigest()


# def file_hasher(path: str) -> str:
#     import os
#     if os.path.isdir(path):
#         raise ValueError('Only file can be hashed')

#     BLOCKSIZE = 65536
#     m = hashlib.sha256()

#     with open(path, 'rb') as fin:
#         buf = fin.read(BLOCKSIZE)
#         while len(buf) > 0:
#             m._update(buf)
#             buf = fin.read(BLOCKSIZE)
#     return m.hexdigest()


def declare_eager_execution():
    import tensorflow as tf
    if not tf.compat.v1.executing_eagerly():
        tf.compat.v1.enable_eager_execution()


def clear_gpu(ind = 0):
    from numba import cuda
    cuda.select_device(ind)
    cuda.close()
    cuda.select_device(ind)
