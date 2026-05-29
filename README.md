# MT Exercise 4

This repository contains my experiments for MT Exercise 4 on Byte Pair Encoding and beam search.

## Language Direction

I used the translation direction:

English → Italian

The training, development, and test data were downloaded with the provided Hugging Face data script.

## Setup

I first created and activated the virtual environment:

```bash
bash scripts/make_virtualenv.sh
source venvs/torch3/bin/activate
```

Then I downloaded the Moses scripts:

```bash
bash scripts/download_moses.sh
```

I also installed the required Python packages:

```bash
pip install subword-nmt pandas matplotlib
```

Because the Hugging Face dataset loader did not work with the initially installed package versions, I used:

```bash
pip uninstall -y datasets huggingface_hub
pip install "datasets==3.6.0" "huggingface_hub==0.24.7"
```

I installed the hotfixed JoeyNMT version in editable mode:

```bash
git clone https://github.com/moritz-steiner/joeynmt-hotfixed.git
cd joeynmt-hotfixed
pip install -e .
cd ..
```

During training, I also had to remove the deprecated `verbose` argument from `ReduceLROnPlateau` in `joeynmt-hotfixed/joeynmt/builders.py`, because it caused a compatibility error with the installed PyTorch version.

## Data Download

I downloaded the English–Italian IWSLT 2017 data with:

```bash
python scripts/download_huggingface_data.py --src en --trg it
```

The script created 100,000 sentence pairs for training.

## Preprocessing

I added the following preprocessing scripts:

```text
scripts/preprocess_tokenize.sh
scripts/preprocess_bpe_2k.sh
scripts/preprocess_bpe_5k.sh
```

First, I tokenized the data with Moses:

```bash
bash scripts/preprocess_tokenize.sh
```

Then I created BPE data with two different vocabulary sizes:

```bash
bash scripts/preprocess_bpe_2k.sh
bash scripts/preprocess_bpe_5k.sh
```

For BPE, I learned joint BPE codes on the concatenated English and Italian training data. I also used vocabulary filtering with language-specific vocabularies and then created a final joint vocabulary file for JoeyNMT.

## Models

I trained three models:

| Model   | BPE | Vocabulary size | Config               |
| ------- | --: | --------------: | -------------------- |
| word_2k |  no |            2000 | configs/word_2k.yaml |
| bpe_2k  | yes |            2000 | configs/bpe_2k.yaml  |
| bpe_5k  | yes |            5000 | configs/bpe_5k.yaml  |

Training was run with:

```bash
bash scripts/train.sh word_2k
bash scripts/train.sh bpe_2k
bash scripts/train.sh bpe_5k
```

Evaluation was run with:

```bash
bash scripts/evaluate.sh word_2k
bash scripts/evaluate.sh bpe_2k
bash scripts/evaluate.sh bpe_5k
```

## BLEU Results

| Model   | BPE | Vocabulary size | BLEU |
| ------- | --: | --------------: | ---: |
| word_2k |  no |            2000 | 11.4 |
| bpe_2k  | yes |            2000 | 20.4 |
| bpe_5k  | yes |            5000 | 21.0 |

The word-level model performed clearly worse than both BPE models. This is expected because the word-level model used a vocabulary limit of only 2000 words, so many rare words were mapped to `<unk>`. The BPE models could split rare words into subword units and therefore preserved more information.

The best model was `bpe_5k`, with a BLEU score of 21.0.

## Manual Translation Inspection

I manually compared the model outputs with the source and reference translations.

A clear difference was that the word-level model produced many `<unk>` tokens. This happened because rare words were outside the limited word vocabulary. For example, in sentences containing rare words or names such as “marshmallow” or “Peter Skillman,” the word-level model often produced `<unk>`, while the BPE models could preserve these words or approximate them with subword units.

The BPE models produced more complete translations and kept more content from the source sentences. The difference between `bpe_2k` and `bpe_5k` was smaller than the difference between the word-level model and the BPE models. Overall, the manual inspection supports the BLEU results: BPE was much better than the word-level vocabulary, and `bpe_5k` was slightly better than `bpe_2k`.

## Beam Size Experiment

For the beam-size experiment, I used the best model, `bpe_5k`.

I added:

```text
scripts/beam_experiment.sh
scripts/plot_beam_results.py
beam_results.csv
beam_size_vs_bleu.png
beam_size_vs_time.png
```

The beam experiment was run with:

```bash
bash scripts/beam_experiment.sh
```

The results were:

| Beam size | BLEU | Time in seconds |
| --------: | ---: | --------------: |
|         1 | 19.5 |              59 |
|         2 | 20.6 |              20 |
|         3 | 21.0 |              28 |
|         4 | 21.1 |              40 |
|         5 | 21.0 |              47 |
|         6 | 21.1 |              54 |
|         7 | 21.0 |              66 |
|         8 | 21.0 |              79 |
|         9 | 21.1 |              96 |
|        10 | 21.1 |              98 |

I created two plots:

```text
beam_size_vs_bleu.png
beam_size_vs_time.png
```

The BLEU score increased from beam size 1 to beam size 4. After beam size 4, the BLEU score stayed almost constant around 21.0–21.1. The generation time generally increased for larger beam sizes. Beam size 1 was unusually slow in my measurement, probably due to runtime noise or system load, but from beam size 2 onward the expected trend is visible.

Based on these results, I would choose beam size 4 in future experiments, because it reaches the best BLEU score while still being much faster than larger beam sizes.

