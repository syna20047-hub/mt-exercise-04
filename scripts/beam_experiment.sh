#!/bin/bash

set -e

MODEL_NAME=bpe_5k
CONFIG=configs/${MODEL_NAME}.yaml
RESULTS=beam_results.csv

echo "beam_size,bleu,time_seconds" > $RESULTS

# Backup original config
cp $CONFIG ${CONFIG}.backup

for BEAM in 1 2 3 4 5 6 7 8 9 10
do
    echo "======================================"
    echo "Running beam size $BEAM"
    echo "======================================"

    # Change beam size in config
    sed -i '' "s/beam_size: .*/beam_size: $BEAM/" $CONFIG

    START=$SECONDS

    OUTPUT=$(bash scripts/evaluate.sh $MODEL_NAME)

    TIME=$((SECONDS - START))

    BLEU=$(echo "$OUTPUT" | grep '"score"' | head -1 | sed 's/[^0-9.]//g')

    echo "$BEAM,$BLEU,$TIME" >> $RESULTS

    echo "$OUTPUT"
    echo "Recorded: beam_size=$BEAM, BLEU=$BLEU, time=$TIME seconds"
done

# Restore original config
mv ${CONFIG}.backup $CONFIG

echo "Beam experiment finished."
echo "Results saved in $RESULTS"
