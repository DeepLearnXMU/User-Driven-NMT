#!/usr/bin/env bash
#train, valid and test
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

HIS=25
CUR=35
DATA="$ONMT/data"
TIME=$(date "+%m%d-%H%M")
OUT="$ONMT/onmt-runs/user-$TIME"
GPUARG="0"
CHECKPOINT=$ONMT/onmt-runs/your_best_model.pt #change it to your best pretrained ckpt


echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test

cp $CHECKPOINT $OUT/models

echo "Step 1: Train"
GPU_OPTS=""
if [[ ! -z "$GPUARG" ]]; then
    GPU_OPTS="-gpu_ranks $GPUARG"
fi
cp $ONMT/config/config-finetune.yml $OUT/config$TIME.yml
CMD="python $ONMT/train.py -save_model $OUT/models/zh-en -train_from $CHECKPOINT -config $ONMT/config/config-finetune.yml -train_history $DATA/cache/topic$HIS.train.src -train_current $DATA/cache/context$CUR.train.src -valid_history $DATA/cache/topic$HIS.valid.src -valid_current $DATA/cache/context$CUR.valid.src -nearby_history $DATA/cache/nearby$HIS.train.src -far_history $DATA/cache/far$HIS.train.src -margin 2 -c_rate 0.5 -contrastive learning> $OUT/train$TIME.log 2>&1"

echo "Training command: $CMD"
echo "Training command: $CMD" > $OUT/cmd$TIME
eval "$CMD"

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
        -test_history $DATA/cache/topic$HIS.test.src -test_current $DATA/cache/context$CUR.test.src \
        -replace_unk  -verbose $GPU_OPTS -batch_size 100 > $OUT/test/test."$MODEL_NAME".log 2>&1

    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/test."$MODEL_NAME".out{,.bpe}
    cat $OUT/test/test."$MODEL_NAME".out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test."$MODEL_NAME".out

    echo "Step 3b: Evaluate BLEU(detok)"
    #test
    echo "$MODEL_NAME" >> $OUT/test/test.lc.bleu
    bleu=$($ONMT/tools/multi-bleu-detok.perl -lc $DATA/test.en < $OUT/test/test."$MODEL_NAME".out)
    echo "$bleu" >> $OUT/test/test.lc.bleu
done

for model in ${arr[@]}
do
    MODEL_NAME=${model##*/}
    echo "For model ${model}:"
    echo "Step 2b: Translate Dev"
    python $ONMT/translate.py -model $model \
        -src $DATA/valid.src \
        -output $OUT/test/valid."$MODEL_NAME".out \
        -test_history $DATA/cache/topic$HIS.valid.src -test_current $DATA/cache/context$CUR.valid.src \
        -replace_unk -verbose $GPU_OPTS -batch_size 100 > $OUT/test/valid."$MODEL_NAME".log 2>&1

    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/valid."$MODEL_NAME".out{,.bpe}
    cat $OUT/test/valid."$MODEL_NAME".out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid."$MODEL_NAME".out

    echo "Step 3b: Evaluate BLEU(detok)"
    #dev
    echo "$MODEL_NAME" >> $OUT/test/valid.lc.bleu
    bleu=$($ONMT/tools/multi-bleu-detok.perl -lc $DATA/valid.en < $OUT/test/valid."$MODEL_NAME".out)
    echo "$bleu" >> $OUT/test/valid.lc.bleu
done