ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

# update these variables
DATA="$ONMT/data"
TIME="0112-1708"
OUT="$ONMT/onmt-runs/user-$TIME"
GPUARG="" # default
GPUARG="0"

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
        -test_history $DATA/user/cache/topic25.test.src -test_current $DATA/user/cache/context35.test.src \
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
        -test_history $DATA/user/cache/topic25.valid.src -test_current $DATA/user/cache/context35.valid.src \
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