
#!/bin/bash

set -e

mkdir -p data/bpe2k

# 1. Learn joint BPE codes on concatenated source + target training data
cat data/tok/train.en data/tok/train.it > data/bpe2k/train.joint

subword-nmt learn-bpe -s 2000 --total-symbols \
    < data/bpe2k/train.joint \
    > data/bpe2k/bpe.codes

# 2. Apply BPE once to training data to build language-specific vocabularies
subword-nmt apply-bpe -c data/bpe2k/bpe.codes \
    < data/tok/train.en \
    > data/bpe2k/train.tmp.en

subword-nmt apply-bpe -c data/bpe2k/bpe.codes \
    < data/tok/train.it \
    > data/bpe2k/train.tmp.it

subword-nmt get-vocab \
    < data/bpe2k/train.tmp.en \
    > data/bpe2k/vocab.en

subword-nmt get-vocab \
    < data/bpe2k/train.tmp.it \
    > data/bpe2k/vocab.it

# 3. Re-apply BPE with vocabulary filtering
subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.en --vocabulary-threshold 50 \
    < data/tok/train.en \
    > data/bpe2k/train.en

subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.it --vocabulary-threshold 50 \
    < data/tok/train.it \
    > data/bpe2k/train.it

subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.en --vocabulary-threshold 50 \
    < data/tok/dev.en \
    > data/bpe2k/dev.en

subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.it --vocabulary-threshold 50 \
    < data/tok/dev.it \
    > data/bpe2k/dev.it

subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.en --vocabulary-threshold 50 \
    < data/tok/test.en \
    > data/bpe2k/test.en

subword-nmt apply-bpe -c data/bpe2k/bpe.codes --vocabulary data/bpe2k/vocab.it --vocabulary-threshold 50 \
    < data/tok/test.it \
    > data/bpe2k/test.it

# 4. Build final joint vocabulary for JoeyNMT
cat data/bpe2k/train.en data/bpe2k/train.it | subword-nmt get-vocab > data/bpe2k/vocab_with_counts.txt
cut -d ' ' -f1 data/bpe2k/vocab_with_counts.txt > data/bpe2k/vocab.txt

rm data/bpe2k/train.tmp.en data/bpe2k/train.tmp.it

echo "BPE 2k preprocessing finished with vocabulary filtering."
