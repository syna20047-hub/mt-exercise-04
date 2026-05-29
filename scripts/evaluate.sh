#!/bin/bash

set -e

scripts=$(dirname "$0")
base=$scripts/..

configs=$base/configs
translations=$base/translations

mkdir -p $translations

src=en
trg=it

num_threads=4

model_name=$1

if [[ -z "$model_name" ]]; then
    echo "Usage: bash scripts/evaluate.sh MODEL_NAME"
    echo "Example: bash scripts/evaluate.sh word_2k"
    echo "Example: bash scripts/evaluate.sh bpe_2k"
    echo "Example: bash scripts/evaluate.sh bpe_5k"
    exit 1
fi


# All models use tokenized test input.
# For BPE models, JoeyNMT applies BPE internally from the config.
data=$base/data/tok

SECONDS=0

echo "################################################################################"
echo "model_name $model_name"
echo "data folder $data"

translations_sub=$translations/$model_name
mkdir -p $translations_sub

raw_out=$translations_sub/test.$model_name.raw.$trg
post_out=$translations_sub/test.$model_name.detok.$trg

# Translate pre-tokenized source input
OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.$src > $raw_out

# Postprocess:
# 1. remove BPE markers if present
# 2. detokenize Italian
sed 's/@@ //g' $raw_out | perl $base/tools/moses-scripts/scripts/tokenizer/detokenizer.perl -l $trg > $post_out

# Compute case-sensitive BLEU against original, non-tokenized reference
cat $post_out | sacrebleu $base/data/test.$trg

echo "time taken:"
echo "$SECONDS seconds"
