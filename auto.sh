#!/bin/bash

sudo rm -rf build dist
sudo docker build -t amir .
sudo docker run -v $(pwd):/lda amir
find $(pwd) | grep whl$
