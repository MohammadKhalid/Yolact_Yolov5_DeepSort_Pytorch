#!/bin/bash
apt update ; apt clean
pip install -r requirements.txt
"$@"