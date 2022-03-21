#!/bin/bash
apt update ; apt clean
apt install ffmpeg libsm6 libxext6 -y
pip install -r requirements.txt
cd yolact_vizta
pip install -r requirements.txt
python setup.py build_ext --inplace
cd ..
"$@"