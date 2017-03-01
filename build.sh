#!/bin/bash

make cython
pip install .
python setup.py bdist_wheel

cd /
python -c "import lda"
