#!/bin/bash
#sudo apt install python3-sphinx
#mkdir docs
#cd docs

#mkdir rst html
#sphinx-quickstart
make html
#cd rst
sphinx-apidoc -o rst ../nefct
cp rst/modules.rst rst/index.rst
#mkdir _static
cp conf.py rst/
sphinx-build -b html ./rst ./html/