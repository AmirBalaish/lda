#!/bin/bash

sudo rm -rf build dist
sudo docker build -t amir .
sudo docker run -v $(pwd):/lda amir
sudo chown -R amir:amir dist/ build/

pip_path=$(find $(pwd) | grep whl$ | head -n 1)
pip uninstall -y lda
pip install $pip_path
