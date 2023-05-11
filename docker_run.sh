#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: docker_run.sh <DATA_DIR> <CKPT_DIR>"
  exit
fi

DATA_DIR=$1
CHECKPOINT_DIR=$2

docker run \
  --rm -it --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v `pwd`:/didi \
  -v $DATA_DIR:/data \
  -v $CHECKPOINT_DIR:/ckpt \
  didi
