#!/bin/sh
source configs_local/mxnet.sh
python3 trainModel.py -f Mxnet -m ssdVgg16 -d ../datasets/fruit -dn fruit