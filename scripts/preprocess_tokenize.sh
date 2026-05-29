#!/bin/bash

set -e

mkdir -p data/tok

MOSES=tools/moses-scripts/scripts/tokenizer/tokenizer.perl

perl $MOSES -l en < data/train.en > data/tok/train.en
perl $MOSES -l it < data/train.it > data/tok/train.it

perl $MOSES -l en < data/dev.en > data/tok/dev.en
perl $MOSES -l it < data/dev.it > data/tok/dev.it

perl $MOSES -l en < data/test.en > data/tok/test.en
perl $MOSES -l it < data/test.it > data/tok/test.it

echo "Tokenization finished."
