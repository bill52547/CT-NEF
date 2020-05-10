import os

from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 0  # 0 for alpha, 1 for beta, 2 for release candicate, 3 for release
MINOR_SUB = 1


def get_version(major, minor, micro, minor_sub):
    from time import time
    from datetime import datetime
    timestamps = int(time())
    short_version = str(major) + '.' + str(minor) + '.' + str(micro) + '.' + str(minor_sub)
    full_version = short_version + '.' + str(timestamps)
    time_string = datetime.fromtimestamp(timestamps).strftime("%Y-%m-%d %H:%M:%S")
    return short_version, full_version, time_string


SHORT_VERSION, FULL_VERSION, TIME_STRING = get_version(MAJOR, MINOR, MICRO, MINOR_SUB)


def write_version_py(filename = 'nefct/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM nefct SETUP.PY
# 
short_version = '%(short_version)s'
full_version = '%(full_version)s'
generated_time = '%(time_string)s'
    
    """
    with open(filename, 'w') as fin:
        fin.write(cnt % {'short_version': SHORT_VERSION,
                         'full_version': FULL_VERSION,
                         'time_string': TIME_STRING})


def run_pytype(out_path = None):
    if out_path is None:
        out_path = os.path.abspath('./.pytype')


write_version_py()

setup(name = 'nefct',
      version = FULL_VERSION,
      py_modules = ['nefct'],
      description = 'Not Enough Functions in CT reconstrctions',
      author = 'Minghao Guo',
      author_email = 'mh.guo0111@gmail.com',
      license = 'Apache',
      packages = find_packages(),
      install_requires = [
          'scipy',
          'matplotlib',
          'h5py',
          'click',
          'numpy',
          'tqdm',
          'numba',
          'deepdish==0.3.6',
      ],
      zip_safe = False,
      entry_points = '''
        [console_scripts]
        nefct=nefct.app.cli:cli
      ''',
      )
