# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: config.py
@date: 4/17/2019
@desc:
'''
import os
import platform

if 'Windows' in platform.system():
    separator = '\\'
else:
    separator = '/'
''' required directories'''
DATABASE_DIR = '/mnt/nfs/NefDatabase/'

# if not os.path.isdir(DATABASE_DIR):
#     raise ValueError(DATABASE_DIR, ' not exists')

RESOURCE_DIR = DATABASE_DIR + 'resources' + separator
CACHE_DIR = DATABASE_DIR + 'caches' + separator
LOG_DIR = DATABASE_DIR + 'logs' + separator
DOC_DIR = DATABASE_DIR + 'docs' + separator
TMP_DIR = DATABASE_DIR + 'tmp' + separator

# if 'TF_USER_OP_PATH' not in os.environ:
#     raise ValueError(
#         'Please declare `TF_USER_OP_PATH` in system env first\n',
#         'e.g. export TF_USER_OP_PATH=/downloads/tensorflow/bazel-bin/tensorflow/core/user_ops/'
#     )

# TF_USER_OP_PATH = os.environ['TF_USER_OP_PATH']

# if 'TF_USER_OP_ROOT' not in os.environ:
#     raise ValueError(
#         'Please declare `TF_USER_OP_ROOT` in system env first\n',
#         'e.g. export TF_USER_OP_ROOT=/downloads/tensorflow/tensorflow/core/user_ops/'
#     )

# TF_USER_OP_ROOT = os.environ['TF_USER_OP_ROOT']

# _PATH_LIST = [DATABASE_DIR, RESOURCE_DIR, CACHE_DIR, LOG_DIR, DOC_DIR, TMP_DIR]

# for _path in _PATH_LIST:
#     if not os.path.isdir(_path):
#         os.mkdir(_path)