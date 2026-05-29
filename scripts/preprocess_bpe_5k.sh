#!/bin/bash

set -e

mkdir -p data/bpe5k

# 1. Learn joint BPE codes on concatenated source + target training data
cat data/tok/train.en data/tok/train.it > data/bpe5k/train.joint

subword-nmt learn-bpe -s 5000 --total-symbols \
    < data/bpe5k/train.joint \
    > data/bpe5k/bpe.codes

# 2. Apply BPE once to training data to build language-specific vocabularies
subword-nmt apply-bpe -c data/bpe5k/bpe.codes \
    < data/tok/train.en \
    > data/bpe5k/train.tmp.en

subword-nmt apply-bpe -c data/bpe5k/bpe.codes \
    < data/tok/train.it \
    > data/bpe5k/train.tmp.it

subword-nmt get-vocab \
    < data/bpe5k/train.tmp.en \
    > data/bpe5k/vocab.en

subword-nmt get-vocab \
    < data/bpe5k/train.tmp.it \
    > data/bpe5k/vocab.it

# 3. Re-apply BPE with vocabulary filtering
subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.en --vocabulary-threshold 50 \
    < data/tok/train.en \
    > data/bpe5k/train.en

subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.it --vocabulary-threshold 50 \
    < data/tok/train.it \
    > data/bpe5k/train.it

subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.en --vocabulary-threshold 50 \
    < data/tok/dev.en \
    > data/bpe5k/dev.en

subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.it --vocabulary-threshold 50 \
    < data/tok/dev.it \
    > data/bpe5k/dev.it

subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.en --vocabulary-threshold 50 \
    < data/tok/test.en \
    > data/bpe5k/test.en

subword-nmt apply-bpe -c data/bpe5k/bpe.codes --vocabulary data/bpe5k/vocab.it --vocabulary-threshold 50 \
    < data/tok/test.it \
    > data/bpe5k/test.it

# 4. Build final joint vocabulary for JoeyNMT
cat data/bpe5k/train.en data/bpe5k/train.it | subword-nmt get-vocab > data/bpe5k/vocab_with_counts.txt
cut -d ' ' -f1 data/bpe5k/vocab_with_counts.txt > data/bpe5k/vocab.txt

rm data/bpe5k/train.tmp.en data/bpe5k/train.tmp.it

echo "BPE 5k preprocessing finished with vocabulary filtering."
