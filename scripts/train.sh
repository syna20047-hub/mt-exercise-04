#!/bin/bash

set -e

# Usage:
# bash scripts/train.sh word_2k
# bash scripts/train.sh bpe_2k
# bash scripts/train.sh bpe_5k

scripts=$(dirname "$0")
base=$scripts/..

num_threads=4

model_name=$1

if [ -z "$model_name" ]; then
    echo "Usage: bash scripts/train.sh MODEL_NAME"
    echo "Example: bash scripts/train.sh word_2k"
    exit 1
fi

logs=$base/logs
mkdir -p $logs
mkdir -p $logs/$model_name

SECONDS=0

OMP_NUM_THREADS=$num_threads python -m joeynmt train $base/configs/$model_name.yaml \
    > $logs/$model_name/out \
    2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"
