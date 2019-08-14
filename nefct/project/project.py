# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: project.py
@date: 5/20/2019
@desc:
'''
from nefct import nef_class
from nefct.geometry.scanner_config import ScannerConfig
from nefct.data.image import Image


@nef_class
class Project:
    mode: str
    scanner: ScannerConfig

    def __call__(self, image: Image):

