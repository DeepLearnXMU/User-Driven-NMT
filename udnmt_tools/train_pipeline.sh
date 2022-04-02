#!/usr/bin/env bash
#train, valid and test
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

# update these variables
DATA="$ONMT/data/wmt17"
TIME=$(date "+%m%d-%H%M")
OUT="$ONMT/onmt-runs/wmt17-$TIME"
GPUARG="0"

echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test

echo "Step 1: Train"
GPU_OPTS=""
if [[ ! -z "$GPUARG" ]]; then
    GPU_OPTS="-gpu_ranks $GPUARG"
fi
CMD="python $ONMT/train.py -save_model $OUT/models/zh-en -config $ONMT/config/config-transformer-base-1GPU.yml > $OUT/train$TIME.log 2>&1"
echo "Training command :: $CMD"
eval "$CMD"

#EOF

# select all models under onmt-runs/run-x/models/
for file in "$OUT"/models/*pt
do
    if test -f "$file"
    then
            arr=(${arr[*]} $file)
    fi
done
if [[ -z "$arr" ]]; then
    echo "Model not found. Looked in $OUT/models/"
    exit 1
fi

GPU_OPTS=""
if [ ! -z "$GPUARG" ]; then
    GPU_OPTS="-gpu $GPUARG"
fi

for model in ${arr[@]}
do
    MODEL_NAME=${model##*/}
    echo "For model ${model}:"
    echo "Step 2a: Translate Test"
    python $ONMT/translate.py -model $model \
        -src $DATA/test.src \
        -output $OUT/test/test."$MODEL_NAME".out \
        -replace_unk  -verbose $GPU_OPTS > $OUT/test/test."$MODEL_NAME".log 2>&1

    echo "Step 2b: Translate Dev"
    python $ONMT/translate.py -model $model \
        -src $DATA/valid.src \
        -output $OUT/test/valid."$MODEL_NAME".out \
        -replace_unk -verbose $GPU_OPTS > $OUT/test/valid."$MODEL_NAME".log 2>&1

    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/test."$MODEL_NAME".out{,.bpe}
    mv $OUT/test/valid."$MODEL_NAME".out{,.bpe}
    cat $OUT/test/valid."$MODEL_NAME".out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid."$MODEL_NAME".out
    cat $OUT/test/test."$MODEL_NAME".out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test."$MODEL_NAME".out

    echo "Step 3b: Evaluate BLEU(detok)"
    #test
    echo "$MODEL_NAME" >> $OUT/test/test.lc.bleu
    bleu=$($ONMT/tools/multi-bleu-detok.perl -lc $DATA/test.tgt < $OUT/test/test."$MODEL_NAME".out)
    echo "$bleu" >> $OUT/test/test.lc.bleu
    #dev
    echo "$MODEL_NAME" >> $OUT/test/valid.lc.bleu
    bleu=$($ONMT/tools/multi-bleu-detok.perl -lc $DATA/valid.tgt < $OUT/test/valid."$MODEL_NAME".out)
    echo "$bleu" >> $OUT/test/valid.lc.bleu

done
#===== EXPERIMENT END ======

